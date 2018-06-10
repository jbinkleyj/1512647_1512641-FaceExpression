
#include "FaceDetection.h"
#include "training.h"

std::map<double, string> mp;

int main(int, char**)
{
		mp[Labels.AN] = "Angry";
		mp[Labels.DI] = "Disgust";
		mp[Labels.FE] = "Fear";
		mp[Labels.HA] = "Happy";
		mp[Labels.NE] = "Neutral";
		mp[Labels.SA] = "Sad";
		mp[Labels.SU] = "Surprise";
	
	MODEL df;
	deserialize("df.dat") >> df;
	std::vector<cv::Rect> rects;
	cv::VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	cv::Mat edges;
	cv::namedWindow("edges", 1);
	for (;;)
	{
			cv::Mat frame;
			cap >> frame; // get a new frame from camera
	
		rects.clear();
		std::vector<cv::Mat> faces = faceDetection.Detect(frame, rects);
					  
		for (int i = 0; i < faces.size(); ++i) {
			double label = predict(faces[i], df);
			cv::rectangle(frame, rects[i], cv::Scalar(0, 0, 0));
			cv::putText(frame, mp[label], cv::Point(rects[i].x, rects[i].y), cv::FONT_HERSHEY_COMPLEX_SMALL
				, 0.8, cvScalar(200, 200, 250));
		}
			imshow("Facial Expression", frame);
			//cv::waitKey(0);
			if (cv::waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}