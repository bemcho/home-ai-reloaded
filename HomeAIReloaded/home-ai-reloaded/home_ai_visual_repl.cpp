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
VideoCapture capture;
String lbp_recognizer_name = "lbphFaceRecognizer.xml";
String clips_vca_rules = "visualcontextrules.clp";


ClipsAdapter clips(clips_vca_rules);

vector<VisualREPL> cameras;
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


	VisualREPL vrepl1("Camera 1", clips, [](Mat f, Mat f_g) {return faceAnnotator.predictWithLBP(f_g);}, true);
	vrepl1.startAt(0, 10);

	VisualREPL vrepl2("Camera 2", clips, [](Mat f, Mat f_g) {return objectsAnnotator.predictWithCAFFE(f, f_g);}, true);
	vrepl2.startAt(1,20);

	VisualREPL vrepl3("Camera 3", clips, [](Mat f, Mat f_g) {return textAnnotator.predictWithTESSERACT(f_g);}, true);
	vrepl3.startAt(2, 10);

	while (true)
	{
		clips.envReset();
		this_thread::sleep_for(std::chrono::milliseconds(50));
		DATA_OBJECT rv;
		clips.envRun();
		clips.envEval("(facts)", rv);
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




