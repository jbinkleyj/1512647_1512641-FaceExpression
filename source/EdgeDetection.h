#pragma once

#include <iostream>
#include <algorithm>
#include <functional>
#include <string>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml.hpp"



class EdgeDetection
{
	int edgeThresh = 1;
	int lowThreshold = 50;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	char* window_name = "Edge Map";

public:
	cv::Mat Detect(cv::Mat &srcImage);

	cv::Mat Crop(const cv::Mat &srcImage, int x, int y, int height, int width);

	cv::Mat Resize(const cv::Mat &srcImage, cv::Size sz);

	EdgeDetection();
	~EdgeDetection();
};

static EdgeDetection edgeDetection;
