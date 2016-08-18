#pragma once
#include <string>
#include "ClipsAdapter.hpp"
#include "VisualContextAnnotator.hpp"
#include  <thread>
#include  "tbb/parallel_invoke.h"
#include "tbb/critical_section.h"
#include <functional>
using namespace std;
namespace hai
{
	class VisualREPL
	{
	public:
		VisualREPL(string replName, ClipsAdapter & aClips, function<vector<Annotation>(cv::Mat, cv::Mat)> annotationFN, bool aShowWindow) : name{ replName }, annotationFN{ annotationFN }, clips{ aClips }, showWindow{ showWindow } {};
		~VisualREPL() { capture.release(); };

		void startAt(int cameraIndex, int framesPerSecond) noexcept
		{
			capture = VideoCapture(cameraIndex);
			if (!capture.isOpened())
			{
				capture.release();
				cout << "--(!)Error opening video capture\nYou do have camera plugged in, right?" << endl;
				return;
			}

			t = std::thread(&VisualREPL::startVisualLoop, this, framesPerSecond);
			t.detach();
		}

		void startAt(const string& streamOrWebCamUrl, int framesPerSecond) noexcept
		{
			capture = VideoCapture(streamOrWebCamUrl);
			if (!capture.isOpened())
			{
				capture.release();
				cout << "--(!)Error opening video capture\nYou do have camera plugged in, right?" << endl;
				return;
			}

			t = std::thread(&VisualREPL::startVisualLoop, this, framesPerSecond);
			t.detach();
		}

	private:
		std::thread t;
		VideoCapture capture;
		bool showWindow;
		string name;
		ClipsAdapter& clips;
		function<vector<Annotation>(cv::Mat, cv::Mat)> annotationFN;

		void startVisualLoop(int framesPerSecond) noexcept
		{
			long fc = 1;

			capture.set(CAP_PROP_FRAME_WIDTH, 10000);
			capture.set(CAP_PROP_FRAME_HEIGHT, 10000);

			capture.set(CAP_PROP_FRAME_WIDTH, (capture.get(CAP_PROP_FRAME_WIDTH) / 2) <= 1280 ? 1280 : capture.get(CAP_PROP_FRAME_WIDTH) / 2);
			capture.set(CAP_PROP_FRAME_HEIGHT, (capture.get(CAP_PROP_FRAME_HEIGHT) / 2) <= 720 ? 720 : capture.get(CAP_PROP_FRAME_HEIGHT) / 2);
			if (showWindow)
			{
				cv::namedWindow(name, WINDOW_OPENGL);
			}

			Mat frame, frame_gray;
			vector<Annotation> annotations;
			capture >> frame;
			while (true)
			{
				if (fc++ % framesPerSecond == 0)
				{
					
					annotations.clear();
					capture >> frame;

					cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
					equalizeHist(frame_gray, frame_gray);
					if (frame.empty())
					{
						cout << " --(!) No captured frame -- Break!" << endl;
						break;
					}

					annotations = annotationFN(frame, frame_gray);
					clips.callFactCreateFN(annotations, name);
				}

				if (showWindow)
				{
					for (auto& annot : annotations)
					{
						rectangle(frame, annot.getRectangle(), CV_RGB(0, 255, 0), 1);
						putText(frame, annot.getDescription(), Point(annot.getRectangle().x, annot.getRectangle().y - 20), CV_FONT_NORMAL, 1.0, CV_RGB(0, 255, 0), 1);
					}
					imshow(name, frame);
				}
				
				if (waitKey(1) == 27)
				{

					break;
				}
			}
			capture.release();
			cv::destroyWindow(name);
		}
	};
}
