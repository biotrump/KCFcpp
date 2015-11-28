#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

extern "C"
{
#include "picornt.h"
}

const char *KCF_win_name="KCF";

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
	VideoCapture vc;
	int index=-1;

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

#if 0
  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;
#endif
	// Frame counter
	int nFrames = 0;

	vc.open(index);
  	//-- 2. Read the video stream
	if(vc.isOpened())
  	{
		//double tick_psec=cv::getTickFrequency();
		int width,height, t_width, t_height,baseline=0;
		int fontFace=FONT_HERSHEY_SIMPLEX;
		int thickness=1;
		double fontScale=1.0;
		unsigned long frame_no=0;
		double fps;

#if 0
		fps = vc.get(CV_CAP_PROP_FPS);
		width = vc.get(CV_CAP_PROP_FRAME_WIDTH);
		height = vc.get(CV_CAP_PROP_FRAME_HEIGHT);

		int num_el = rows*cols;
		int len = num_el*frame.elemSize1();
#endif

#define	MAX_OBJECTS	(3)
		int nfaces=0;
		int face_init=0;
		// Tracker results
		Rect result;

		for(;;){
			vc >> frame; 	//get one frame
	      	if(  !frame.empty() ){
				Mat frame_gray;
				height = frame.rows;
				width = frame.cols;

				flip(frame,frame,1);
				frame_no++;
				cvtColor( frame, frame_gray, CV_BGR2GRAY );
				uchar *grey_buf = frame_gray.ptr();

				//flip(frame_gray,frame_gray,1);
				int max_faces=MAX_OBJECTS;
				float rs[MAX_OBJECTS],cs[MAX_OBJECTS],ss[MAX_OBJECTS];
				if(!face_init){
					nfaces=pico_facedetection(FaceDetection_PICO/*FaceDetection_RYAN*/,
								grey_buf, width, height, max_faces, rs, cs, ss, 1.0f);
					printf("nfacecs=%d\n", nfaces);
					if(nfaces){
						Rect roi[nfaces];
						for(int i = 0; i < nfaces; i++){
							roi[i].width= ss[i]*1.2/1.414f;
							roi[i].height= ss[i]*1.6/1.414f;
							roi[i].x =cs[i] - roi[i].width/2;
							roi[i].y =rs[i] - roi[i].height/2 + roi[i].height/12;
							rectangle( frame, Point( roi[i].x, roi[i].y ),
									Point( roi[i].x+roi[i].width, roi[i].y+roi[i].height),
									Scalar( 0, 255, 0 ), 2, 8 );
							rectangle( frame_gray, Point( roi[i].x, roi[i].y ),
									Point( roi[i].x+roi[i].width, roi[i].y+roi[i].height),
									Scalar( 0, 255, 0 ), 2, 8 );
						}
						face_init=1;
						tracker.init( Rect(roi[0].x, roi[0].y, roi[0].width, roi[0].height),
									  frame );
					}else{
						face_init=0;
					}
				}else{
					result = tracker.update(frame);
					rectangle( frame, Point( result.x, result.y ),
							   Point( result.x+result.width, result.y+result.height),
							   Scalar( 0, 0, 255 ), 2, 8 );
					rectangle( frame_gray, Point( result.x, result.y ),
							   Point( result.x+result.width, result.y+result.height),
							   Scalar( 0, 0, 255 ), 2, 8 );
				}

				//fine tune the delay to fixed fps as the video file's original fps.
				Size textSize = cv::getTextSize(cv::format("%d", frame_no), fontFace, fontScale,thickness, &baseline);
				t_height = textSize.height;
				t_width = textSize.width;
				putText(frame, cv::format("%4.1f", fps), Point(0,t_height), fontFace, fontScale,cv::Scalar(0,0,255),thickness);
				putText(frame, cv::format("%d", frame_no), Point(width-t_width,t_height), fontFace, fontScale,cv::Scalar(0,0,255),thickness);
				imshow(KCF_win_name, frame);
				imshow("KCF Gray",frame_gray);
			}else{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			char c = waitKey(10);
			if( c == 27 || c == 'q' || c == 'Q' ) { break; }
			if(c == 'r' ) face_init=0;	//the tracking object is missing, find face again!
		}//for
	}

#if 0
	while ( getline(listFramesFile, frameName) ){
		frameName = frameName;

		// Read each frame from the list
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
			result = tracker.update(frame);
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(1);
		}
	}
#endif
}
