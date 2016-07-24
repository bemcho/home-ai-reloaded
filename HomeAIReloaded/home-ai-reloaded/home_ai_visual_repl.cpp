#include "opencv2/dnn.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Annotation.h"
#include "tesseract/baseapi.h"
#include "VisualContextAnnotator.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_invoke.h"
#include "clips/clips.h"

#include <fstream>
#include <iostream>
#include <cstdlib>

using namespace std;


String face_cascade_name = "cascade_frontalface.xml";
String window_name = "Capture - Face detection";
VideoCapture capture;
String lbp_recognizer_name = "lbphFaceRecognizer.xml";
String clips_vca_rules = "visualcontextrules.clp";

Ptr<void> theCLIPSEnv;

DATA_OBJECT rv;
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

void AddDetectFact2(void *environment, string type, Rect at, string ontology)
{
	void *newFact;
	void *templatePtr;
	void *theMultifield;
	DATA_OBJECT theValue;
	/*============================================================*/
	/* Disable garbage collection. It's only necessary to disable */
	/* garbage collection when calls are made into CLIPS from an */
	/* embedding program. It's not necessary to do this when the */
	/* the calls to user code are made by CLIPS (such as for */
	/* user-defined functions) or in the case of this example, */
	/* there are no calls to functions which can trigger garbage */
	/* collection (such as Send or FunctionCall). */
	/*============================================================*/
	//IncrementGCLocks(environment);
	/*==================*/
	/* Create the fact. */
	/*==================*/
	templatePtr = EnvFindDeftemplate(environment, "visualdetect");
	newFact = EnvCreateFact(environment, templatePtr);
	if (newFact == NULL) return;
	/*==============================*/
	/* Set the value of the type slot. */
	/*==============================*/
	theValue.type = SYMBOL;
	theValue.value = EnvAddSymbol(environment, type.c_str());
	EnvPutFactSlot(environment, newFact, "type", &theValue);
	/*==============================*/
	/* Set the value of the z slot. */
	/*==============================*/
	theMultifield = EnvCreateMultifield(environment, 4);
	SetMFType(theMultifield, 1, INTEGER);
	SetMFValue(theMultifield, 1, EnvAddLong(environment, at.x));

	SetMFType(theMultifield, 2, INTEGER);
	SetMFValue(theMultifield, 2, EnvAddLong(environment, at.y));

	SetMFType(theMultifield, 3, INTEGER);
	SetMFValue(theMultifield, 3, EnvAddLong(environment, at.width));

	SetMFType(theMultifield, 4, INTEGER);
	SetMFValue(theMultifield, 4, EnvAddLong(environment, at.height));


	SetDOBegin(theValue, 1);
	SetDOEnd(theValue, 4);
	theValue.type = MULTIFIELD;
	theValue.value = theMultifield;
	EnvPutFactSlot(environment, newFact, "at", &theValue);
	/*==============================*/
	/* Set the value of the what slot. */
	/*==============================*/
	theValue.type = SYMBOL;
	stringstream onto;
	onto << "\"" << ontology << "\"";
	theValue.value = EnvAddSymbol(environment, onto.str().c_str());
	EnvPutFactSlot(environment, newFact, "ontology", &theValue);
	/*=================================*/
	/* Assign default values since all */
	/* slots were not initialized. */
	/*=================================*/
	EnvAssignFactSlotDefaults(environment, newFact);
	/*==========================================================*/
	/* Enable garbage collection. Each call to IncrementGCLocks */
	/* should have a corresponding call to DecrementGCLocks. */
	/*==========================================================*/
	//EnvDecrementGCLocks(environment);
	/*==================*/
	/* Assert the fact. */
	/*==================*/
	EnvAssert(environment, newFact);
}
/**
* @function main
*/
int main(int, char**)
{
	theCLIPSEnv = CreateEnvironment();
	EnvLoad(theCLIPSEnv, clips_vca_rules.c_str());
	char * cs = "(deftemplate visualdetect"
		" (slot type (default object))"
		" (multislot at)"
		" (slot ontology)"
		" )";
	EnvBuild(theCLIPSEnv, cs);

	textAnnotator.loadTESSERACTModel("..\\Release/", "eng", tesseract::OEM_TESSERACT_CUBE_COMBINED);

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
	CAFFERect = Rect((capture.get(CAP_PROP_FRAME_WIDTH) / 2) - 250, (capture.get(CAP_PROP_FRAME_HEIGHT) / 2) - 250, 300, 300);
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

		{

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

		}

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

		if (fc % 30 == 0)
		{
			/*vector<Annotation> localAnnotations;
			for (int i = 0;i < 3 && i < objectsDetects.size();i++)
			{
				localAnnotations.push_back(objectsAnnotator.predictWithCAFFEInRectangle(objectsDetects[i], frame));
			}

			caffeAnnotations.clear();
			caffeAnnotations = localAnnotations;*/

		}

		vector<Annotation> allAnnotations;
		allAnnotations = lbpAnnotations;
		allAnnotations.insert(allAnnotations.end(), textAnnotations.begin(), textAnnotations.end());
		allAnnotations.insert(allAnnotations.end(), caffeAnnotations.begin(), caffeAnnotations.end());

		EnvReset(theCLIPSEnv);
		//define new facts here
		stringstream  fact;
		for (auto& annot : allAnnotations)
		{
			AddDetectFact2(theCLIPSEnv, annot.getType(), annot.getRectangle(), annot.getDescription());

			rectangle(frame, annot.getRectangle(), CV_RGB(0, 255, 0), 1);
			putText(frame, annot.getDescription(), Point(annot.getRectangle().x, annot.getRectangle().y - 20), CV_FONT_NORMAL, 1.0, CV_RGB(0, 255, 0), 1);

		}

		if (objectContours.size() > 0)
		{
			for (auto c : objectContours)
			{
				AddDetectFact2(theCLIPSEnv, "contour", boundingRect(Mat(c)), "contour");
			}

			drawContours(frame, objectContours, -1, CV_RGB(255, 213, 21), 2);
		}
		EnvRun(theCLIPSEnv, -1);
		EnvEval(theCLIPSEnv, "(facts)", &rv);
		imshow(window_name, frame);
		fc++;
		//-- bail out if escape was pressed
		if (waitKey(1) == 27)
		{
			break;
		}

	}

	DestroyEnvironment(theCLIPSEnv);

	exit(0);
}




