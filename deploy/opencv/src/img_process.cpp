#include <iostream>
#include <chrono>
#include "cv_utils.hpp"

using namespace std;
using namespace cv;

std::vector<ImageData> img_crop(std::vector<ImageData> images)
{ 
    vector<ImageData> results;
    cout << "开始裁剪图像..." << endl;

    try
    {
        // 初始化裁剪器
        ImageCropper cropper;

        // 批量固定区域裁剪
        ImageCropper::Config cfg(
            ImageCropper::FIXED_ROI, 
            cv::Rect(100, 50, 2048, 2048), // x,y,width,height
            true                          // 自动调整边界
        );

        // 记录开始时间点
        auto start = std::chrono::high_resolution_clock::now();

        results = cropper.batchProcess(images, cfg);

        // 记录结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        // 计算耗时（转换为毫秒）
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "裁图耗时: " << duration.count() << " 毫秒" << std::endl;
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
    }

    return results;
}

std::vector<ImageData> img_resize(std::vector<ImageData> images, cv::Size target_size)
{
    vector<ImageData> results;
    cout << "开始缩放图像..." << endl;

    try
    {
        // 创建缩放器实例
        ImageResizer resizer;

        // 配置缩放参数
        ImageResizer::Config cfg;
        cfg.strategy = ImageResizer::EXACT_SIZE;
        cfg.mode = ImageResizer::LINEAR; // 使用双线性插值
        cfg.target_size = target_size;

        // 记录开始时间点
        auto start = std::chrono::high_resolution_clock::now();

        results = resizer.batchProcess(images, cfg);

        // 记录结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        // 计算耗时（转换为毫秒）
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "缩放耗时: " << duration.count() << " 毫秒" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return results;
}

// 图像预处理函数
vector<ImageData> preprocess_image(const std::string &folder_path, int target_height, int target_width) {
    vector<ImageData> images = loadAllImages(folder_path);
    vector<ImageData> processed;

    // 调整尺寸
    if (target_height > 0 && target_width > 0) {
        processed = img_resize(images, cv::Size(target_width, target_height));
    }

    for (auto &img : processed) {
        // 转换为三通道
        if (img.image.channels() != 3) {
            cv::cvtColor(img.image, img.image, cv::COLOR_GRAY2BGR);
        }

        // 转换为 float 并归一化
        img.image.convertTo(img.image, CV_32F);
        img.image = (img.image / 255.0f - 0.5f) / 0.5f;
    }

    return processed;
}