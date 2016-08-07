#include "VisualContextAnnotator.hpp"

namespace hai
{
	VisualContextAnnotator::VisualContextAnnotator()
	{
		model = face::createLBPHFaceRecognizer();
		tess = make_unique<tesseract::TessBaseAPI>();
		net = make_unique<dnn::Net>();
		cascade_classifier = make_unique<CascadeClassifier>();

	}


	VisualContextAnnotator::~VisualContextAnnotator()
	{
		/*cascade_classifier.release();
		model.release();
		net.release();
		tess.release();*/
	}

	void VisualContextAnnotator::loadCascadeClassifier(const string cascadeClassifierPath)
	{
		//-- 1. Load the cascade
		if (!cascade_classifier->load(cascadeClassifierPath) || cascade_classifier->empty()) { printf("--(!)Error loading face cascade\n"); };

	}

	void VisualContextAnnotator::loadLBPModel(const string path, double maxDistance)
	{
		model->load(path);
		this->maxDistance = maxDistance;
	}

	void VisualContextAnnotator::loadTESSERACTModel(const string& dataPath, const string& lang, tesseract::OcrEngineMode mode)
	{
		tess->Init(dataPath.c_str(), lang.c_str(), tesseract::OEM_TESSERACT_CUBE_COMBINED);
	}

	void VisualContextAnnotator::loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath)
	{
		Ptr<dnn::Importer> importer;
		try                                     //Try to import Caffe GoogleNet model
		{
			importer = dnn::createCaffeImporter(modelProtoTextPath, modelBinPath);
		}
		catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
		{
			std::cerr << err.msg << std::endl;
		}
		if (!importer)
		{
			std::cerr << "Can't load network by using the following files: " << std::endl;
			std::cerr << "prototxt:   " << modelProtoTextPath << std::endl;
			std::cerr << "caffemodel: " << modelBinPath << std::endl;
			std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
			std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		}

		importer->populateNet(*net);

		importer.release();
		classNames = readClassNames(synthWordPath);
	}

	void VisualContextAnnotator::detectWithCascadeClassifier(vector<Rect>& result, const Mat & frame_gray, Size minSize)noexcept
	{
		cascade_classifier->detectMultiScale(frame_gray, result, 1.1, 3, 0, minSize, Size());
	}

	void VisualContextAnnotator::detectWithMorphologicalGradient(vector<Rect>& result, const Mat & frame_gray, Size minSize, Size kernelSize) noexcept
	{
		/**http://stackoverflow.com/questions/23506105/extracting-text-opencv**/

		{
			// morphological gradient
			Mat grad;
			Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
			morphologyEx(frame_gray, grad, MORPH_GRADIENT, morphKernel);
			// binarize
			Mat bw;
			threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
			// connect horizontally oriented regions
			Mat connected;
			morphKernel = getStructuringElement(MORPH_RECT, kernelSize);
			morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
			// find contours
			Mat mask = Mat::zeros(bw.size(), CV_8UC1);
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
			if (contours.size() == 0)
			{
				return;
			}
			// filter contours
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
			{
				Rect rect = boundingRect(contours[idx]);
				Mat maskROI(mask, rect);
				maskROI = Scalar(0, 0, 0);
				// fill the contour
				drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
				// ratio of non-zero pixels in the filled region
				double r = (double)countNonZero(maskROI) / (rect.width*rect.height);

				if (r > .45 /* assume at least 45% of the area is filled if it contains text */
					&&
					(rect.height > minSize.height && rect.width > minSize.width) /* constraints on region size */
														/* these two conditions alone are not very robust. better to use something
														like the number of significant peaks in a horizontal projection as a third condition */
					)
				{
					result.push_back(rect);
				}
			}
		}
	}
	void VisualContextAnnotator::detectObjectsWithCanny(vector<vector<Point>>& contours, const Mat& frame_gray, double lowThreshold, Size minSize) noexcept
	{
		Mat detected_edges;
		/// Reduce noise with a kernel 3x3
		//blur(frame_gray, detected_edges, Size(3, 3));
		GaussianBlur(frame_gray, detected_edges, Size(3, 3), 1);


		/// Canny detector
		Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, 3);
		Mat connected;
		Mat morphKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(detected_edges, connected, MORPH_CLOSE, morphKernel);
		// connect horizontally oriented regions
		vector<vector<Point>> localContours;
		vector<Vec4i> hierarchy;
		findContours(connected, localContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		for (auto cnt : localContours)
		{
			double epsilon = 0.01*arcLength(cnt, true);
			vector<Point> approx;
			approxPolyDP(cnt, approx, epsilon, true); //only closed curves
			if (approx.size() > 0)
			{
				Rect r = boundingRect(approx);
				if (r.size().width >= minSize.width && r.size().height >= minSize.height)
				{
					contours.push_back(cnt);
				}
			}

		}
	}

	void VisualContextAnnotator::detectObjectsWithCanny(vector<Rect>& result, const  Mat & frame_gray, double lowThreshold, Size minSize) noexcept
	{

		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		detectObjectsWithCanny(contours, frame_gray, lowThreshold, minSize);

		for (auto cnt : contours)
		{
			Rect r = boundingRect(cnt);
			result.push_back(r);
		}
	}

	Annotation VisualContextAnnotator::predictWithLBPInRectangle(const Rect& detect, const  Mat& frame_gray) noexcept
	{
		Mat face = frame_gray(detect);
		int predictedLabel = -1;
		double confidence = 0.0;
		try
		{
			model->predict(face, predictedLabel, confidence);
		}
		catch (...)
		{

		}

		std::stringstream fmt;
		if (predictedLabel > 0 && confidence <= maxDistance)
		{
			fmt << model->getLabelInfo(predictedLabel) << "L:" << predictedLabel << "C:" << confidence;
		}
		else
		{
			fmt << "Unknown Human" << "L:" << predictedLabel << "C:" << confidence;
		}
		return Annotation(detect, fmt.str(), "human");
	}

	struct PredictWithLBPBody {
		VisualContextAnnotator & vca_;
		vector<Rect> detects_;
		Mat frame_gray_;
		vector<Annotation>& result_;
		PredictWithLBPBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray, const size_t tsize) :vca_(u), detects_(detects), frame_gray_(frame_gray), result_(vector<Annotation>(tsize)) {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
				result_.push_back(vca_.predictWithLBPInRectangle(detects_[i], frame_gray_));
		}
	};


	void VisualContextAnnotator::predictWithLBP(vector<Annotation>& annotations, const Mat& frame_gray) noexcept
	{
		static tbb::affinity_partitioner affinityLBP;

		vector<Rect> detects;
		detectWithCascadeClassifier(detects, frame_gray);

		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;
		PredictWithLBPBody parallelLBP(*this, detects, frame_gray, tsize);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelLBP,                    // Body of loop
			affinityLBP);

		annotations = parallelLBP.result_;
	}

	void VisualContextAnnotator::predictWithLBP(vector<Annotation>& annotations, const  vector<Rect> detects, const Mat & frame_gray) noexcept
	{
		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;
		static tbb::affinity_partitioner affinityLBP2;
		PredictWithLBPBody parallelLBP(*this, detects, frame_gray, tsize);

		vector<Annotation> result(tsize);
		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelLBP,                    // Body of loop
			affinityLBP2);

		annotations = parallelLBP.result_;
	}

	Annotation VisualContextAnnotator::predictWithCAFFEInRectangle(const Rect & detect, const  Mat & frame) noexcept
	{
		cv::Mat img;
		img = Scalar::all(0);

		resize(frame(detect), img, Size(244, 244));

		tbb::critical_section cs;

		cs.lock();
		dnn::Blob inputBlob;
		dnn::Blob prob;
		inputBlob = dnn::Blob(img);
		//Convert Mat to dnn::Blob image batch
		net->setBlob(".data", inputBlob);        //set the network input
		net->forward();                          //compute output
		prob = net->getBlob("prob");
		int classId;
		double classProb;
		getMaxClass(prob, classId, classProb);//find the best class
		stringstream caffe_fmt = stringstream();
		caffe_fmt << "P: " << classProb * 100 << "%" << std::endl;
		caffe_fmt << "Class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
		// critical section here
		cs.unlock();
		return Annotation(detect, caffe_fmt.str(), classNames.at(classId));
	}

	struct PredictWithCAFFEBody {
		VisualContextAnnotator & vca_;
		vector<Rect> detects_;
		Mat frame_gray_;
		vector<Annotation>& result_;
		PredictWithCAFFEBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray, size_t tsize) :vca_(u), detects_(detects), frame_gray_(frame_gray), result_(vector<Annotation>(tsize)) {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				result_.push_back(vca_.predictWithCAFFEInRectangle(detects_[i], frame_gray_));
			}
		}
	};
	void VisualContextAnnotator::predictWithCAFFE(vector<Annotation>& annotations, const Mat & frame, const  Mat & frame_gray) noexcept
	{
		static tbb::affinity_partitioner affinityDNN2;
		vector<Rect> detects;
		detectObjectsWithCanny(detects, frame_gray);
		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;
		PredictWithCAFFEBody parallelDNN(*this, detects, frame, tsize);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelDNN,                    // Body of loop
			affinityDNN2);

		annotations = vector<Annotation>(parallelDNN.result_);
	}

	void VisualContextAnnotator::predictWithCAFFE(vector<Annotation>& annotations, const vector<Rect> detects, const Mat & frame) noexcept
	{
		static tbb::affinity_partitioner affinityDNN;

		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;
		PredictWithCAFFEBody parallelDNN(*this, detects, frame, tsize);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelDNN,                    // Body of loop
			affinityDNN);

		annotations = vector<Annotation>(parallelDNN.result_);
	}

	/* Find best class for the blob (i. e. class with maximal probability) */
	void VisualContextAnnotator::getMaxClass(dnn::Blob& probBlob, int& classId, double& classProb)
	{
		Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
		Point classNumber;
		minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
		classId = classNumber.x;
	}

	std::vector<String> VisualContextAnnotator::readClassNames(const string filename = "synset_words.txt")
	{
		std::vector<String> classNames;
		std::ifstream fp(filename);
		if (!fp.is_open())
		{
			std::cerr << "File with classes labels not found: " << filename << std::endl;
			exit(-1);
		}
		std::string name;
		while (!fp.eof())
		{
			std::getline(fp, name);
			if (name.length())
				classNames.push_back(name.substr(name.find(' ') + 1));
		}
		fp.close();
		return classNames;
	}


	Annotation VisualContextAnnotator::predictWithTESSERACTInRectangle(const Rect & detect, const Mat & frame_gray) noexcept
	{
		static tbb::critical_section cs;
		cs.lock();
			Mat sub = frame_gray(detect).clone();
			tess->SetImage((uchar*)sub.data, sub.size().width, sub.size().height, sub.channels(), sub.step1());
			int result = tess->Recognize(0);
			string strText(tess->GetUTF8Text());
		cs.unlock();
		if (result == 0)
		{
			return Annotation(detect, strText.substr(0, strText.size() - 2), "text");
		}
		else
		{
			return Annotation(detect, "object", "contour");
		}
		
	}
	struct PredictWithTESSERACTBody {
		VisualContextAnnotator & vca_;
		vector<Rect> detects_;
		Mat frame_gray_;
		vector<Annotation>& result_;
		PredictWithTESSERACTBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray, const size_t tsize) : vca_(u), detects_(detects), frame_gray_(frame_gray), result_(vector<Annotation>(tsize)) {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				result_.push_back(vca_.predictWithTESSERACTInRectangle(detects_[i], frame_gray_));
			}
		}
	};

	void VisualContextAnnotator::predictWithTESSERACT(vector<Annotation>& annotations, const Mat & frame_gray) noexcept
	{
		static tbb::affinity_partitioner affinityTESSERACT;

		vector<Rect> detects;
		detectWithMorphologicalGradient(detects, frame_gray);
		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;
		PredictWithTESSERACTBody parallelTESSERACT(*this, detects, frame_gray, tsize);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelTESSERACT,                    // Body of loop
			affinityTESSERACT);

		annotations = vector<Annotation>(parallelTESSERACT.result_);
	}

	void VisualContextAnnotator::predictWithTESSERACT(vector<Annotation>& annotations, const vector<Rect> detects, const Mat & frame_gray) noexcept
	{
		static tbb::affinity_partitioner affinityTESSERACT2;

		const size_t tsize = detects.size();
		if (tsize <= 0)
			return;

		PredictWithTESSERACTBody parallelTESSERACT(*this, detects, frame_gray, tsize);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, tsize), // Index space for loop
			parallelTESSERACT,                    // Body of loop
			affinityTESSERACT2);

		annotations = vector<Annotation>(parallelTESSERACT.result_);
	}
}