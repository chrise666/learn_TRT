#ifndef CV_API_H
#define CV_API_H

#include "cv_utils.hpp"

std::vector<ImageData> img_crop(std::vector<ImageData> images);
std::vector<ImageData> img_resize(std::vector<ImageData> images, cv::Size target_size);

std::vector<ImageData> preprocess_image(const std::string &folder_path, int target_height, int target_width);

#endif