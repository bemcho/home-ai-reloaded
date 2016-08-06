#pragma once
#include<string>
#include <vector>

#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.hpp"
#include "tesseract/baseapi.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/critical_section.h"
using namespace  std;
using namespace cv;
namespace hai
{
	class VisualContextAnnotator
	{
	public:
		VisualContextAnnotator();
		virtual ~VisualContextAnnotator();
		void loadCascadeClassifier(const string cascadeClassifierPath);
		void loadLBPModel(const string path, double maxDistance = 65.0);
		void loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath);
		void loadTESSERACTModel(const string& dataPath, const string& lang, tesseract::OcrEngineMode mode = tesseract::OEM_DEFAULT);

		void detectWithCascadeClassifier(vector<Rect>& result,const Mat& frame_gray, Size minSize = Size(80, 80)) noexcept;
		void detectWithMorphologicalGradient(vector<Rect>& result, const Mat& frame, Size minSize = Size(8, 8), Size kernelSize = Size(9, 1)) noexcept;
		void detectObjectsWithCanny(vector<Rect>& result, const Mat& frame_gray, double lowThreshold = 77, Size minSize = Size(80, 80)) noexcept;
		void detectObjectsWithCanny(vector<vector<Point>>& result, const Mat& frame_gray, double lowThreshold = 77, Size minSize = Size(80, 80)) noexcept;

		void predictWithLBP(vector<Annotation>& annotations, const Mat & frame_gray) noexcept;
		void predictWithLBP(vector<Annotation>& annotations, const vector<Rect> detects, const Mat & frame_gray) noexcept;
		Annotation predictWithLBPInRectangle(const Rect & detect, const Mat & frame_gray) noexcept;

		void predictWithCAFFE(vector<Annotation>& annotations, const Mat & frame, const Mat & frame_gray) noexcept;
		void predictWithCAFFE(vector<Annotation>& annotations, const vector<Rect> detects, const Mat & frame) noexcept;
		Annotation predictWithCAFFEInRectangle(const Rect & detect, const Mat & frame)noexcept;

		void predictWithTESSERACT(vector<Annotation>& annotations, const cv::Mat & frame_gray) noexcept;
		void predictWithTESSERACT(vector<Annotation>& annotations, const vector<Rect> detects, const cv::Mat & frame_gray) noexcept;
		Annotation predictWithTESSERACTInRectangle(const Rect & detect, const Mat & frame_gray) noexcept;

	private:
		Ptr<face::FaceRecognizer> model;
		unique_ptr<CascadeClassifier> cascade_classifier;
		unique_ptr<tesseract::TessBaseAPI> tess;
		unique_ptr<dnn::Net> net;

		double maxDistance;
		void getMaxClass(dnn::Blob & probBlob, int & classId, double & classProb);
		std::vector<String> readClassNames(const string filename);
		std::vector<String> classNames;
	};
}
