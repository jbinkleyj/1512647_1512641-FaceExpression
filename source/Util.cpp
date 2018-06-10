#include "Util.h"

bool build_feature(cv::Mat & faceImage, std::vector<sample_type>& samples)
{

	cv::Mat resizeImage = edgeDetection.Resize(faceImage, cv::Size(60, 60));
	//imshow("a", resizeImage);
	cv::Mat edgeImage = edgeDetection.Detect(resizeImage);

	/*cv::imshow("edge Image", edgeImage);
	/cv::waitKey(0);*/
	int nrows = edgeImage.rows;
	int ncols = edgeImage.cols;
	//cout << "nrows : " << nrows << endl;
	//cout << "ncols : " << ncols << endl;
	std::vector<cv::Point> white_point;

	for (int y = 0; y < nrows; ++y) {
		uchar *prow = edgeImage.ptr<uchar>(y);
		for (int x = 0; x < ncols; ++x) {
			if ((uchar)prow[x] != 0)
				white_point.push_back(cv::Point(x, y));
		}
	}

	//std::cout << "diem trang " << white_point.size() << endl;
	// tính euledean distance từ một điểm tới điểm trắng gàn nhất
	bool flag = true;
	sample_type m;
	int index = 0;
	for (int y = 0; y < nrows; ++y) {
		uchar *prow = edgeImage.ptr<uchar>(y);
		for (int x = 0; x < ncols; ++x) {
			double min_dist = 10000.0;
			for (int i = 0; i < white_point.size(); ++i) {
				if (x != white_point[i].x && y != white_point[i].y)
					min_dist = std::min(min_dist, euclidean_dist(white_point[i], cv::Point(x, y)));
			}

			if (min_dist >= 10000.0 - 10) {
				flag = false;
				break;
			}

			m(index++) = min_dist;
		}
	}

	//cout << "index : " << index << endl;
	if (flag)
		samples.push_back(m);
	return flag;
}

void get_training_data(char *path, std::vector<sample_type>& samples, std::vector<double>& labels)
{

	// lấy tên các bức ảnh 
	std::vector<string> ListOfFileName;
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(path)) != NULL) {
		while ((ent = readdir (dir)) != NULL)
		{
			ListOfFileName.push_back(string(ent->d_name));
		}
	}

	else {
		perror("");
		return;
	}

	/*cout << "all file : " << endl;
	for (int i = 0; i < ListOfFileName.size(); ++i) {
		cout << ListOfFileName[i] << "\\";
	}

	cout << endl;*/
	string path_str(path);
	//cout << path << endl;
	for (int i = 0; i < ListOfFileName.size(); ++i) {

		//cout << path_str + ListOfFileName[i] << endl;;
		cv::Mat image = cv::imread(path_str + ListOfFileName[i]);


		if (!image.data) {
			continue;
		}

		//cv::imshow("ab", faceImage);
		//cv::waitKey(0);
		//cv::imshow("a", image);
		// gọi hàm build feature

		// lấy mặt
		cv::Mat rimage;
		cv::resize(image, rimage, cv::Size(256, 256));
		std::vector<cv::Rect> rects;
		std::vector<cv::Mat> faces = faceDetection.Detect(rimage, rects);
		//std::cout << ListOfFileName[i] << endl;
		bool flag = true;
		if (faces.empty()) {
			flag = build_feature(image, samples);
		}
		else {
			flag = build_feature(faces[0], samples);
		}
		// get label của image 

		if (!flag)
			continue;

		string s = ListOfFileName[i].substr(POS_OF_LABEL, LENGTH_OF_LABEL);
		
		if (s == Expression.AN)
			labels.push_back(Labels.AN);
		else if (s == Expression.DI)
			labels.push_back(Labels.DI);
		else if (s == Expression.FE)
			labels.push_back(Labels.FE);
		else if (s == Expression.HA)
			labels.push_back(Labels.HA);
		else if (s == Expression.NE)
			labels.push_back(Labels.NE);
		else if (s == Expression.SA)
			labels.push_back(Labels.SA);
		else if (s == Expression.SU)
			labels.push_back(Labels.SU);
		else {
			cv::waitKey(0);
			cerr << "Khong co nhan!";
			return;
		}
	}

}

double euclidean_dist(cv::Point p1, cv::Point p2)
{
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}
