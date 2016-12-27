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
		cascade_classifier.release();
		model.release();
		net.release();
		tess.release();
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
		this->lbpModelPath = path;
	}

	void VisualContextAnnotator::loadTESSERACTModel(const string& dataPath, const string& lang, tesseract::OcrEngineMode mode)
	{
		tess->Init(dataPath.c_str(), lang.c_str(), tesseract::OEM_TESSERACT_CUBE_COMBINED);
	}

	void VisualContextAnnotator::train(vector<cv::Mat> samples, int label, string ontology) noexcept
	{
		tbb::mutex::scoped_lock(training);
		vector<int> labels(samples.size(), label);
		model->train(samples, labels);
		model->setLabelInfo(label, ontology);
	}

	void VisualContextAnnotator::update(vector<cv::Mat> samples, int label, string ontology) noexcept
	{
		tbb::mutex::scoped_lock(lbpTrainingLock);
		vector<int> labels(samples.size(), label);
		model->update(samples, labels);
		model->setLabelInfo(label, ontology);
		model->save(lbpModelPath);
		model->load(lbpModelPath);
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
	vector<Rect>  VisualContextAnnotator::detectWithCascadeClassifier(const Mat frame_gray, Size minSize)noexcept
	{
		tbb::mutex::scoped_lock lck(cascadeClassLock);
		vector<Rect> result;
		Mat frame_gray_local(frame_gray);
		cascade_classifier->detectMultiScale(frame_gray_local, result, 1.1, 10, 0, minSize, Size());
		return result;
	}

	vector<Rect> VisualContextAnnotator::detectWithMorphologicalGradient(const Mat frame_gray, Size minSize, Size kernelSize) noexcept
	{
		tbb::mutex::scoped_lock lck(morphGradientLock);
		vector<Rect> result;
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
				return result;
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
		return result;
	}
	vector<Annotation> VisualContextAnnotator::detectContoursWithCanny(const Mat frame_gray, double lowThreshold, Size minSize) noexcept
	{
		tbb::mutex::scoped_lock lck(contoursWithCannyLock);
		vector<Annotation> result;
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

		for (auto& cnt : localContours)
		{
			double epsilon = 0.01*arcLength(cnt, true);
			vector<Point> approx;
			approxPolyDP(cnt, approx, epsilon, true); //only closed curves
			if (approx.size() > 0)
			{
				Rect r = boundingRect(approx);
				if (r.size().width >= minSize.width && r.size().height >= minSize.height)
				{
					result.push_back(Annotation(cnt, "contour of " + std::to_string(cnt.size()) + " points.", "contour"));
				}
			}
		}
		return result;
	}

	vector<Rect> VisualContextAnnotator::detectObjectsWithCanny(const  Mat  frame_gray, double lowThreshold, Size minSize) noexcept
	{
		tbb::mutex::scoped_lock lck(objectsWithCannyLock);
		vector<Rect> result;

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

		for (auto& cnt : localContours)
		{
			double epsilon = 0.01*arcLength(cnt, true);
			vector<Point> approx;
			approxPolyDP(cnt, approx, epsilon, true); //only closed curves
			if (approx.size() > 0)
			{
				Rect r = boundingRect(approx);
				if (r.size().width >= minSize.width && r.size().height >= minSize.height)
				{
					result.push_back(r);
				}
			}
		}
		return result;
	}

	Annotation VisualContextAnnotator::predictWithLBPInRectangle(const Rect detect, const  Mat frame_gray, const string& annotationType) noexcept
	{
		tbb::mutex::scoped_lock lck(lbpInRectLock);
		Mat face = frame_gray(detect);
		int predictedLabel = -1;
		double confidence = 0.0;

		model->predict(face, predictedLabel, confidence);

		std::stringstream fmt;
		if (predictedLabel > 0 && confidence <= maxDistance)
		{
			fmt << model->getLabelInfo(predictedLabel) << "L:" << predictedLabel << "C:" << confidence;
		}
		else
		{
			fmt << "Unknown " << annotationType << "L:" << predictedLabel << "C:" << confidence;
		}
		return Annotation(detect, fmt.str(), annotationType);
	}

	struct PredictWithLBPBody {
		const string annotationType;
		VisualContextAnnotator & vca_;
		vector<Rect> detects_;
		Mat frame_gray_;
		vector<Annotation>& result_;
		PredictWithLBPBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray, vector<Annotation>& result, const string aAnnotationType) :vca_{ u }, detects_{ detects }, frame_gray_{ frame_gray }, result_{ result }, annotationType{ aAnnotationType } {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
				result_.push_back(vca_.predictWithLBPInRectangle(detects_[i], frame_gray_, annotationType));
		}
	};


	vector<Annotation> VisualContextAnnotator::predictWithLBP(const Mat frame_gray) noexcept
	{
		tbb::mutex::scoped_lock lck(lbp1Lock);
		static tbb::affinity_partitioner affinityLBP;

		vector<Rect> detects = detectWithCascadeClassifier(frame_gray);
		
		if (detects.size() <= 0)
			return vector<Annotation>();

		vector<Annotation> annotations(detects.size());
		PredictWithLBPBody parallelLBP(*this, detects, frame_gray, annotations, "human");

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelLBP,                    // Body of loop
			affinityLBP);

		return annotations;
	}

	vector<Annotation>  VisualContextAnnotator::predictWithLBP(const  vector<Rect> detects, const Mat frame_gray, const string& annotationType) noexcept
	{
		tbb::mutex::scoped_lock lck(lbp2Lock);
		static tbb::affinity_partitioner affinityLBP;

		if (detects.size() <= 0)
			return vector<Annotation>();

		vector<Annotation> annotations(detects.size());
		PredictWithLBPBody parallelLBP(*this, detects, frame_gray, annotations, annotationType);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelLBP,                    // Body of loop
			affinityLBP);

		return annotations;
	}

	Annotation VisualContextAnnotator::predictWithCAFFEInRectangle(const Rect detect, const  Mat frame) noexcept
	{
		tbb::mutex::scoped_lock lck(caffeInRectLock);

		cv::Mat img;
		img = Scalar::all(0);
		resize(frame(detect), img, Size(244, 244));

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
		caffe_fmt << "N:" << '\'' << classNames.at(classId) << '\'' << " P:" << classProb * 100 << "%" << std::endl;
		caffe_fmt << " ID:" << classId << std::endl;
		// critical section here
		return Annotation(detect, caffe_fmt.str(), classNames.at(classId));
	}

	struct PredictWithCAFFEBody {
		VisualContextAnnotator & vca_;
		vector<Rect> detects_;
		Mat frame_gray_;
		vector<Annotation>& result_;
		PredictWithCAFFEBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray) :vca_(u), detects_(detects), frame_gray_(frame_gray), result_(vector<Annotation>(detects.size())) {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				result_.push_back(vca_.predictWithCAFFEInRectangle(detects_[i], frame_gray_));
			}
		}
	};
	vector<Annotation> VisualContextAnnotator::predictWithCAFFE(const Mat frame, const  Mat  frame_gray) noexcept
	{
		tbb::mutex::scoped_lock lck(caffe1Lock);
		static tbb::affinity_partitioner affinityDNN2;
		vector<Rect> detects = detectObjectsWithCanny(frame_gray);

		if (detects.size() <= 0)
			return vector<Annotation>();

		PredictWithCAFFEBody parallelDNN(*this, detects, frame);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelDNN,                    // Body of loop
			affinityDNN2);

		return parallelDNN.result_;
	}

	vector<Annotation> VisualContextAnnotator::predictWithCAFFE(const vector<Rect> detects, const Mat frame) noexcept
	{
		tbb::mutex::scoped_lock lck(caffe2Lock);
		static tbb::affinity_partitioner affinityDNN;

		if (detects.size() <= 0)
			return vector<Annotation>();

		PredictWithCAFFEBody parallelDNN(*this, detects, frame);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelDNN,                    // Body of loop
			affinityDNN);

		return parallelDNN.result_;
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


	Annotation VisualContextAnnotator::predictWithTESSERACTInRectangle(const Rect  detect, const Mat frame_gray) noexcept
	{
		tbb::mutex::scoped_lock lck(tessInRectLock);
		Mat sub = frame_gray(detect).clone();
		if (detect.height < 50)
		{
			resize(sub, sub, Size(detect.width*3 , detect.height * 3));
		}
		
		
		tess->SetImage((uchar*)sub.data, sub.size().width, sub.size().height, sub.channels(), sub.step1());
		int result = tess->Recognize(0);
		string strText(tess->GetUTF8Text());
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
		PredictWithTESSERACTBody(VisualContextAnnotator & u, const vector<Rect> detects, const Mat frame_gray) : vca_(u), detects_(detects), frame_gray_(frame_gray), result_(vector<Annotation>(detects.size())) {}
		void operator()(const tbb::blocked_range<size_t>& range) const {
			for (size_t i = range.begin(); i != range.end(); ++i)
			{
				result_.push_back(vca_.predictWithTESSERACTInRectangle(detects_[i], frame_gray_));
			}
		}
	};

	vector<Annotation> VisualContextAnnotator::predictWithTESSERACT(const Mat frame_gray) noexcept
	{
		tbb::mutex::scoped_lock lck(tess1Lock);
		static tbb::affinity_partitioner affinityTESSERACT;

		vector<Rect> detects = detectWithMorphologicalGradient(frame_gray);
		if (detects.size() <= 0)
			return vector<Annotation>();

		PredictWithTESSERACTBody parallelTESSERACT(*this, detects, frame_gray);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelTESSERACT,                    // Body of loop
			affinityTESSERACT);

		return  parallelTESSERACT.result_;
	}

	vector<Annotation> VisualContextAnnotator::predictWithTESSERACT(const vector<Rect> detects, const Mat frame_gray) noexcept
	{
		tbb::mutex::scoped_lock lck(tess2Lock);
		static tbb::affinity_partitioner affinityTESSERACT2;

		if (detects.size() <= 0)
			return vector<Annotation>();

		PredictWithTESSERACTBody parallelTESSERACT(*this, detects, frame_gray);

		tbb::parallel_for(tbb::blocked_range<size_t>(0, detects.size()), // Index space for loop
			parallelTESSERACT,                    // Body of loop
			affinityTESSERACT2);

		return parallelTESSERACT.result_;
	}
}