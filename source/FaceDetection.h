#pragma once

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/opencv/cv_image.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml.hpp"

using namespace dlib;
using namespace std;

class FaceDetection {

public:
	FaceDetection() {
		deserialize(data) >> sp;
		detector = get_frontal_face_detector();
	}
	~FaceDetection() {

	}
	std::vector<cv::Mat> Detect(const  cv::Mat &srcImg, std::vector<cv::Rect>& rects) {
		// We need a face detector.  We will use this to get bounding boxes for
		// each face in an image.
		

		// And we also need a shape_predictor.  This is the tool that will predict face
		// landmark positions given an image and face bounding box.  Here we are just
		// loading the model from the shape_predictor_68_face_landmarks.dat file you gave
		// as a command line argument.
		


		//cout << "processing image " << endl;
		dlib::array2d<bgr_pixel> img;
		dlib::assign_image(img, dlib::cv_image<bgr_pixel>(srcImg));

		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		std::vector<rectangle> dets = detector(img);
		//cout << "Number of faces detected: " << dets.size() << endl;

		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		std::vector<full_object_detection> shapes;
		//////////// Tuan Kyou
		std::vector<std::vector<cv::Point2i> >  arrPointList;
		////////////
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			full_object_detection shape = sp(img, dets[j]);
			/*cout << "number of parts: " << shape.num_parts() << endl;
			cout << "pixel position of first part:  " << shape.part(0) << endl;
			cout << "pixel position of second part: " << shape.part(1) << endl;*/
			// You get the idea, you can get all the face part locations if
			// you want them.  Here we just store them in shapes so we can
			// put them on the screen.
			shapes.push_back(shape);
			////////////// Tuan Kyou
			std::vector<cv::Point2i> PointList;
			for (int h = 0; h < shape.num_parts(); h++) {
				if (shape.part(h).x() < 0 || shape.part(h).y() < 0 ||
					shape.part(h).y() >= srcImg.cols || shape.part(h).x() >= srcImg.rows)
					continue;

				cv::Point2i p(shape.part(h).x(), shape.part(h).y());
				PointList.push_back(p);
			}
			arrPointList.push_back(PointList);
			////////////// End Tuan Kyou
		}

		//////////////////// Tuan Kyou
		std::vector<cv::Mat> arrMat;
		std::vector<cv::Rect> arrRect;
		for (int i = 0; i < arrPointList.size(); i++) {
			cv::Rect r = cv::boundingRect(arrPointList[i]);
			rects.push_back(r);
			cv::Mat crop = srcImg(r);
			arrMat.push_back(crop);
		}

		// tìm tọa đồ top left, bottom right
		////////////////////
		return arrMat;
	}
private:
	std::string data = "sp.dat";
	shape_predictor sp;
	frontal_face_detector detector;
protected:
};

static FaceDetection faceDetection;