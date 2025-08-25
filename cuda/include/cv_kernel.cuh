#ifndef CV_KERNEL_CUH
#define CV_KERNEL_CUH

#include <opencv2/opencv.hpp>


cv::Mat cudaCrop(const cv::Mat &src, const cv::Rect &roi);


#endif
