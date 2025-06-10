#include <iostream>
#include <chrono>
#include "image_utils.hpp"

using namespace std;
using namespace cv;

int img_process()
{
    // 图像路径
    const string img_path = "C:\\Users\\ADMIN\\Desktop\\xxxx";
    const string save_path = "C:\\Users\\ADMIN\\Desktop\\xxxx\\res";
    
    cout << "Start loading images..." << endl;

    try
    {
        // 初始化裁剪器
        ImageCropper cropper;

        // 加载原始图像
        vector<ImageData> all_images = loadAllImages(img_path);
        cout << "Successfully loaded " << all_images.size() 
             << " images" << endl;

        // 批量固定区域裁剪
        ImageCropper::Config cfg(
            ImageCropper::FIXED_ROI, 
            cv::Rect(100, 50, 2048, 2048), // x,y,width,height
            true                          // 自动调整边界
        );

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
        return -1;
    }

    return 0;
}
