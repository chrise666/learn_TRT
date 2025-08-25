#ifndef CV_API_H
#define CV_API_H

#include "cv_utils.hpp"

bool img_crop(const std::string img_path, const std::string save_path, ImageCropper::Config &cfg);
bool img_resize(const std::string img_path, const std::string save_path, cv::Size target_size);

#endif