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
#include "tbb/mutex.h"
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

		vector<Rect>  detectWithCascadeClassifier(const Mat frame_gray, Size minSize = Size(80, 80)) noexcept;
		vector<Rect>  detectWithMorphologicalGradient(const Mat frame, Size minSize = Size(8, 8), Size kernelSize = Size(9, 1)) noexcept;
		vector<Rect>  detectObjectsWithCanny(const Mat frame_gray, double lowThreshold = 77, Size minSize = Size(80, 80)) noexcept;
		vector<Annotation>  detectContoursWithCanny(const Mat frame_gray, double lowThreshold = 77, Size minSize = Size(80, 80)) noexcept;

		vector<Annotation> predictWithLBP(const Mat  frame_gray) noexcept;
		vector<Annotation> predictWithLBP(const vector<Rect> detects, const Mat  frame_gray) noexcept;
		Annotation predictWithLBPInRectangle(const Rect  detect, const Mat  frame_gray) noexcept;

		vector<Annotation> predictWithCAFFE(const Mat frame, const Mat frame_gray) noexcept;
		vector<Annotation> predictWithCAFFE(const vector<Rect> detects, const Mat frame) noexcept;
		Annotation predictWithCAFFEInRectangle(const Rect detect, const Mat frame)noexcept;

		vector<Annotation> predictWithTESSERACT(const cv::Mat frame_gray) noexcept;
		vector<Annotation> predictWithTESSERACT(const vector<Rect> detects, const cv::Mat frame_gray) noexcept;
		Annotation predictWithTESSERACTInRectangle(const Rect detect, const Mat frame_gray) noexcept;

	private:
		Ptr<face::FaceRecognizer> model;
		unique_ptr<CascadeClassifier> cascade_classifier;
		unique_ptr<tesseract::TessBaseAPI> tess;
		unique_ptr<dnn::Net> net;
		tbb::mutex m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, lbpInRect, tessInRect, caffeInRect;

		double maxDistance;
		void getMaxClass(dnn::Blob & probBlob, int & classId, double & classProb);
		std::vector<String> readClassNames(const string filename);
		std::vector<String> classNames;
	};
}
