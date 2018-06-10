#include "EdgeDetection.h"



cv::Mat EdgeDetection::Detect(cv::Mat &srcImage)
{
	cv::Mat dst, detected_edges, blurImage;
	blur(srcImage, blurImage, cv::Size(3, 3));

	Canny(blurImage, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	dst = cv::Scalar::all(0);
	srcImage.copyTo(dst, detected_edges);
	//cv::imshow(window_name, dst);
	//cv::waitKey(0);
	return dst;
}

cv::Mat EdgeDetection::Crop(const cv::Mat & srcImage,int x, int y, int height, int width)
{
	// rgb -> grayscale
	cv::Mat grayImage;
	if (srcImage.type() == CV_8UC1)
		cvtColor(srcImage, grayImage, CV_RGB2GRAY);
	else
		grayImage = srcImage.clone();

	cv::Rect croppedRectagle = cv::Rect(x, y, height, width);
	cv::Mat croppedImage = grayImage(croppedRectagle);

	return croppedImage;
}

cv::Mat EdgeDetection::Resize(const cv::Mat & srcImage, cv::Size sz)
{
	cv::Mat dst;
	resize(srcImage, dst, sz, 0, 0, cv::INTER_CUBIC);

	return dst;
}

EdgeDetection::EdgeDetection()
{

}


EdgeDetection::~EdgeDetection()
{
}
