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
#include "tbb/mutex.h"
#include "ClipsAdapter.hpp"
#include "VisualREPL.hpp"
#include "ClipsAdapter.hpp"

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>

using namespace std;
using namespace hai;

String face_cascade_name = "cascade_frontalface.xml";
String window_name = "Home AI";
String lbp_recognizer_name = "lbphFaceRecognizer.xml";
String clips_vca_rules = "visualcontextrules.clp";


ClipsAdapter clips(clips_vca_rules);

vector<shared_ptr<VisualREPL>> cameras;
VisualContextAnnotator faceAnnotator;
VisualContextAnnotator textAnnotator;
VisualContextAnnotator objectsAnnotator;

thread textAnnotatorTrainerThread;
tbb::mutex trainMutex;
int lowThreshold = 77;
const int MAX_CAMERAS = 5;
const bool WINDOW_SHOW = true;

vector<Annotation> annotateFaceContoursFN(Mat f, Mat f_g)
{
	vector<Annotation> result;
	vector<Annotation> face;
	vector<Annotation> contours;
	vector<Annotation> objects;

	tbb::parallel_invoke(
		[&]
	()
	{
		objects = objectsAnnotator.predictWithLBP(textAnnotator.detectObjectsWithCanny(f_g), f_g, "object");
	},
		[&]
	()
	{
		face = faceAnnotator.predictWithLBP(f_g);
	},
		[&]
	()
	{
		contours = objectsAnnotator.detectContoursWithCanny(f_g);
	}
	);
	result.insert(result.end(), face.begin(), face.end());
	result.insert(result.end(), objects.begin(), objects.end());
	result.insert(result.end(), contours.begin(), contours.end());

	return result;
}

vector<Annotation> annotateObjectsFN(Mat f, Mat f_g)
{
	vector<Annotation> result;
	vector<Annotation> face;
	vector<Annotation> objects;
	vector<Annotation> contours;
	vector<Annotation> texts;
	
 	tbb::parallel_invoke(
		[&]
	()
	{
		texts = textAnnotator.predictWithTESSERACT(f_g);
	},
		[&]
	()
	{
		objects = objectsAnnotator.predictWithLBP(textAnnotator.detectObjectsWithCanny(f_g), f_g, "object");
	},

		[&]
	()
	{
		face = faceAnnotator.predictWithLBP(f_g);
	},
		[&]
	()
	{
		contours = objectsAnnotator.detectContoursWithCanny(f_g);
	}
	);
	result.insert(result.end(), texts.begin(), texts.end());
	result.insert(result.end(), objects.begin(), objects.end());
	result.insert(result.end(), face.begin(), face.end());
	result.insert(result.end(), contours.begin(), contours.end());

	return result;
}
vector<Annotation> annotateTextContoursFN(Mat f, Mat f_g)
{
	vector<Annotation> result;
	vector<Annotation> objects;
	vector<Annotation> contours;

	tbb::parallel_invoke(
		[&]
	()
	{
		objects = textAnnotator.predictWithTESSERACT(f_g);

	},
		[&]
	()
	{
		contours = objectsAnnotator.detectContoursWithCanny(f_g);
	}
	);
	result.insert(result.end(), objects.begin(), objects.end());
	result.insert(result.end(), contours.begin(), contours.end());

	return result;
}
void updateLBPModelFN(vector<cv::Mat> samples, int  label, string ontology, bool& aTInProgress)
{
	tbb::mutex::scoped_lock(trainMutex);
	cout << "Calling LBP model -> update\n";
	aTInProgress = true;
	objectsAnnotator.update(samples, label, ontology);
	aTInProgress = false;
}
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

	for (int i = 0; i < MAX_CAMERAS; i++)
	{
		shared_ptr<VisualREPL> vreplP;
		if (i == 0)
		{
			vreplP = make_shared<VisualREPL>(VisualREPL("Stream " + std::to_string(i), clips, annotateObjectsFN, updateLBPModelFN, WINDOW_SHOW));
		}
		else if (i == 1)
		{
			vreplP = make_shared<VisualREPL>(VisualREPL("Stream " + std::to_string(i), clips, annotateFaceContoursFN, updateLBPModelFN, WINDOW_SHOW));
		}
		else if (i == 2)
		{
			vreplP = make_shared<VisualREPL>(VisualREPL("Stream " + std::to_string(i), clips, annotateTextContoursFN, updateLBPModelFN, WINDOW_SHOW));
		}
		else
		{
			vreplP = make_shared<VisualREPL>(VisualREPL("Stream " + std::to_string(i), clips, annotateFaceContoursFN, updateLBPModelFN, WINDOW_SHOW));
		}

		if (vreplP->startAt(i, 30))
		{
			cameras.push_back(vreplP);
			cout << "--(!)Camera found on " << i << " device index." << endl;
		}
		else
		{
			vreplP.reset();
		}
	}

	while (true)
	{

		this_thread::sleep_for(std::chrono::milliseconds(100));
		//DATA_OBJECT rv;
		//clips.envEval("(cyclicCallback)", rv);
		clips.envRun();
		this_thread::sleep_for(std::chrono::milliseconds(50));
		//-- bail out if escape was pressed
		if (waitKey(1) == 27)
		{
			break;
		}

	}
	std::terminate();

	return EXIT_SUCCESS;
}




