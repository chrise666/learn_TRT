#include <iostream>
#include <chrono>
#include "cv_utils.hpp"

using namespace std;
using namespace cv;

bool img_crop(const string img_path, const string save_path, ImageCropper::Config &cfg)
{ 
    cout << "Start loading images..." << endl;

    try
    {
        // 初始化裁剪器
        ImageCropper cropper;

        // 加载原始图像
        vector<ImageData> all_images = loadAllImages(img_path);
        cout << "Successfully loaded " << all_images.size() 
             << " images" << endl;

        // 记录开始时间点
        auto start = std::chrono::high_resolution_clock::now();

        auto result = cropper.batchProcess(all_images, cfg);

        // 记录结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        // 计算耗时（转换为毫秒）
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "代码运行时间: " 
                << duration.count() << " 毫秒"
                << std::endl;

        // 保存结果
        saveResults(result, save_path, "_centercrop");
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return false;
    }

    return true;
}

bool img_resize(const string img_path, const string save_path, cv::Size target_size)
{
    try {
        // 加载图像
        cv::Mat image = cv::imread(img_path);
        if (image.empty()) {
            throw std::runtime_error("无法加载图像");
        }
        
        std::cout << "原始尺寸: " << image.size() << "\n";
        
        // 图像尺寸调整
        cv::Mat result = ImageResizer::smartResize(image, target_size);
        std::cout << "调整为: " << target_size << "\n";
        cv::imwrite(save_path, result);
    }
    catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return false;
    }

    return true;
}