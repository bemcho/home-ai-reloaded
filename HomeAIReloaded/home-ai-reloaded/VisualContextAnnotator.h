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
#include "Annotation.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/critical_section.h"
using namespace  std;
using namespace cv;

class VisualContextAnnotator
{
public:
	VisualContextAnnotator();
	virtual ~VisualContextAnnotator();
	void loadCascadeClassifier(const string cascadeClassifierPath);
	void loadLBPModel(const string path, double maxDistance = 65.0);
	void loadCAFFEModel(const string modelBinPath, const string modelProtoTextPath, const string synthWordPath);
	void detectWithCascadeClassifier(vector<Rect>& result, Mat& frame_gray, Size minSize = Size(80, 80));
	void detectWithMorphologicalGradient(vector<Rect>& result, Mat& frame, Size minSize = Size(8, 8), Size kernelSize = Size(9, 1));
	void detectObjectsWithCanny(vector<Rect>& result, Mat& frame, double lowThreshold = 77, Size minSize = Size(80, 80));
	void detectObjectsWithCanny(vector<vector<Point>>& result, Mat& frame, double lowThreshold = 77, Size minSize = Size(80, 80));
	Annotation predictWithLBPInRectangle(const Rect & detect, Mat & frame_gray);
	void predictWithLBP(vector<Annotation>& annotations, cv::Mat & frame_gray);
	void predictWithLBP(vector<Annotation>& annotations, vector<Rect> detects, cv::Mat & frame);
	void predictWithCAFFE(vector<Annotation>& annotations, cv::Mat & frame, cv::Mat & frame_gray);
	void predictWithCAFFE(vector<Annotation>& annotations, vector<Rect> detects, cv::Mat & frame);
	Annotation predictWithCAFFEInRectangle(const Rect & detect, Mat & frame);

private:
	CascadeClassifier cascade_classifier;
	Ptr<face::FaceRecognizer> model;
	double maxDistance;
	dnn::Net net;
	
	void getMaxClass(dnn::Blob & probBlob, int * classId, double * classProb);
	std::vector<String> readClassNames(const string filename);
	std::vector<String> classNames;
};

