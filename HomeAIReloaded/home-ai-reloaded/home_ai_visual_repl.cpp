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
int lowThreshold = 77;
const int MAX_CAMERAS = 5;
const bool WINDOW_SHOW = true;

vector<Annotation> annotateFaceTextContoursFN(Mat f, Mat f_g)
{
	vector<Annotation> result;
	vector<Annotation> face;
	vector<Annotation> text;
	vector<Annotation> contours;

	tbb::parallel_invoke(
		[&]
	()
	{
		text =  textAnnotator.predictWithTESSERACT(f_g);
	},
		[&]
	()
	{
		face =  faceAnnotator.predictWithLBP(f_g) ;
	},
		[&]
	()
	{
		contours = faceAnnotator.detectContoursWithCanny(f_g);
	}
	);
	result.insert(result.end(), text.begin(), text.end());
	result.insert(result.end(), face.begin(), face.end());
	result.insert(result.end(), contours.begin(), contours.end());

	return result;
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
		cout << "--(!)Camera found on " << i << " device index." << endl;
		shared_ptr<VisualREPL> vreplP = make_shared<VisualREPL>(VisualREPL("Stream " + std::to_string(i), clips, annotateFaceTextContoursFN, WINDOW_SHOW));
		if (vreplP->startAt(i, 10))
		{
			cameras.push_back(vreplP);
		}
		else
		{
			vreplP.reset();
		}
	}

	while (true)
	{
		DATA_OBJECT rv;
		this_thread::sleep_for(std::chrono::milliseconds(100));
		//clips.envEval("(facts)", rv);
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




