//#include "FaceDetection.h"
//#include "training.h"
//
//std::map<double, string> mp;
//
//int main(int argc, char **argv) {
//	mp[Labels.AN] = "Angry";
//	mp[Labels.DI] = "Disgust";
//	mp[Labels.FE] = "Fear";
//	mp[Labels.HA] = "Happy";
//	mp[Labels.NE] = "Neutral";
//	mp[Labels.SA] = "Sad";
//	mp[Labels.SU] = "Surprise";
//
//
//		MODEL df;
//		deserialize("df.dat") >> df;
//		cout << df.get_binary_decision_functions().size() << endl;
//		std::vector<cv::Rect> rects;
//		cv::Mat image = cv::imread(argv[1]);
//		if (!image.data)
//			return -1;
//		cv::Mat resizeImage = edgeDetection.Resize(image, cv::Size(255, 255));
//	
//		std::vector<cv::Mat> faces = faceDetection.Detect(resizeImage, rects);
//	
//		double tmp;
//		if (faces.empty())
//			tmp = predict(image, df);
//		else
//			tmp  = predict(faces[0], df);
//		cout << tmp << endl;
//		cout << mp[tmp] << endl;
//	//training(argv[1]);
//
//	return 0;
//}