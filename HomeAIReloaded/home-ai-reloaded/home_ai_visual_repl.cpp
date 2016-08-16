#include "opencv2/dnn.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.hpp"
#include "tesseract/baseapi.h"
#include "VisualContextAnnotator.hpp"
#include "tbb/blocked_range.h"
#include "tbb/parallel_invoke.h"
#include "ClipsAdapter.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>

using namespace std;
using namespace hai;

String face_cascade_name = "cascade_frontalface.xml";
String window_name = "Capture - Face detection";
VideoCapture capture;
String lbp_recognizer_name = "lbphFaceRecognizer.xml";
String clips_vca_rules = "visualcontextrules.clp";


ClipsAdapter clips(clips_vca_rules);

Rect_<int> CAFFERect;
Mat frame, frame_gray;
bool drawFaceDetectsToWindow = true;
bool drawEdgeDetectsToWindow = false;
bool detectAndDisplayWithCAFFEConf = true;
stringstream caffe_fmt;
vector<Annotation> lbpAnnotations;
vector<Annotation> textAnnotations;
vector<Rect> textDetects;
vector<Rect> objectsDetects;
vector<vector<Point>> objectContours;
vector<Vec4i> objectHierachies;
vector<Rect> faceDetects;
vector<Annotation> caffeAnnotations;
VisualContextAnnotator faceAnnotator;
VisualContextAnnotator textAnnotator;
VisualContextAnnotator objectsAnnotator;
int lowThreshold = 77;


/**
* @function main
*/
int main(int, char**)
{

	textAnnotator.loadTESSERACTModel("..\\Release/", "eng");

	String modelTxt = "bvlc_googlenet.prototxt";
	String modelBin = "bvlc_googlenet.caffemodel";

	faceAnnotator.loadCascadeClassifier(face_cascade_name);
	faceAnnotator.loadLBPModel(lbp_recognizer_name);

	objectsAnnotator.loadCAFFEModel(modelBin, modelTxt, "synset_words.txt");
	objectsAnnotator.loadLBPModel(lbp_recognizer_name);

	cv::namedWindow(window_name, WINDOW_OPENGL);
	//tesseract init
	//--1.5. Init Camera
	for (int i = 0; i < 500; i++)
	{
		capture = VideoCapture(i);
		if (!capture.isOpened())
		{
			capture.release();
			cout << "--(!)Error opening video capture\nYou do have camera plugged in, right?" << endl;
			if (i == 49)
				return -1;

			continue;
		}
		else
		{
			cout << "--(!)Camera found on " << i << " device index.";
			break;
		}
	}

	capture.set(CAP_PROP_FRAME_WIDTH, 10000);
	capture.set(CAP_PROP_FRAME_HEIGHT, 10000);

	capture.set(CAP_PROP_FRAME_WIDTH, (capture.get(CAP_PROP_FRAME_WIDTH) / 2) <= 1280 ? 1280 : capture.get(CAP_PROP_FRAME_WIDTH) / 2);
	capture.set(CAP_PROP_FRAME_HEIGHT, (capture.get(CAP_PROP_FRAME_HEIGHT) / 2) <= 720 ? 720 : capture.get(CAP_PROP_FRAME_HEIGHT) / 2);
	CAFFERect = Rect((capture.get(CAP_PROP_FRAME_WIDTH) / 2.0) - 250, (capture.get(CAP_PROP_FRAME_HEIGHT) / 2.0) - 250, 300, 300);
	long fc = 6;

	while (true)
	{

		capture >> frame;

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		if (frame.empty())
		{
			cout << " --(!) No captured frame -- Break!" << endl;
			break;
		}


		tbb::parallel_invoke(
			[]
		{
			vector<Annotation> localAnnotations;
			faceDetects.clear();
			faceAnnotator.predictWithLBP(localAnnotations, frame_gray);
			lbpAnnotations.clear();
			lbpAnnotations = localAnnotations;
		},

			[]
		{
			vector<Annotation> localAnnotations;
			textAnnotator.predictWithTESSERACT(localAnnotations, frame_gray);
			textAnnotations.clear();
			textAnnotations = localAnnotations;
		}
		);


		{
			vector<vector<Point>> localContours;
			vector<Rect> localDetects;


			objectsAnnotator.detectObjectsWithCanny(localContours, frame_gray, lowThreshold, Size(50, 50));
			objectContours.clear();
			objectContours = localContours;

			for (auto& cnt : localContours)
			{
				localDetects.push_back(boundingRect(cnt));
			}
			objectsDetects.clear();
			objectsDetects = localDetects;
		}

		/*if (fc % 30 == 0)
		{
			vector<Annotation> localAnnotations;
			for (int i = 0;i < 3 && i < objectsDetects.size();i++)
			{
				localAnnotations.push_back(objectsAnnotator.predictWithCAFFEInRectangle(objectsDetects[i], frame));
			}

			caffeAnnotations.clear();
			caffeAnnotations = localAnnotations;
		}*/

		vector<Annotation> allAnnotations;
		allAnnotations = lbpAnnotations;
		allAnnotations.insert(allAnnotations.end(), textAnnotations.begin(), textAnnotations.end());
		allAnnotations.insert(allAnnotations.end(), caffeAnnotations.begin(), caffeAnnotations.end());

		clips.envReset();
		DATA_OBJECT rv;
		//define new facts here
		stringstream  fact;
		tbb::parallel_invoke(
			[&]
		{
			clips.callFactCreateFN(allAnnotations);

			if (objectContours.size() > 0)
			{
				for (auto& c : objectContours)
				{
					clips.callFactCreateFN(Annotation(boundingRect(Mat(c)), "contour", "contour"));
				}

				
			}
		},

			[&]
		{
			for (auto& annot : allAnnotations)
			{
				rectangle(frame, annot.getRectangle(), CV_RGB(0, 255, 0), 1);
				putText(frame, annot.getDescription(), Point(annot.getRectangle().x, annot.getRectangle().y - 20), CV_FONT_NORMAL, 1.0, CV_RGB(0, 255, 0), 1);
			}
		},
			[]
		{
			drawContours(frame, objectContours, -1, CV_RGB(255, 213, 21), 2);
		}
		);
		

		
		clips.envRun();
		clips.envEval("(facts)", rv);
		imshow(window_name, frame);
		fc++;
		//-- bail out if escape was pressed
		if (waitKey(1) == 27)
		{
			break;
		}
	}
	

	return EXIT_SUCCESS;
}




