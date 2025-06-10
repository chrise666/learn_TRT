#include "image_utils.hpp"
#include <filesystem>
#include <unordered_set>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;


// 加载单张图片
Mat loadImage(const string &img_path, bool show_img)
{
    // 检查文件是否存在
    if (!filesystem::exists(img_path))
    {
        throw runtime_error("File not found: " + img_path);
    }

    // 读取图像
    Mat image = imread(img_path);
    if (image.empty())
    {
        throw runtime_error("Failed to load image: " + img_path);
    }

    // 显示图像
    if (show_img)
    {
        namedWindow("Image", WINDOW_AUTOSIZE); // 创建窗口
        imshow("Image", image);
        waitKey(0);             // 等待用户按键
        destroyWindow("Image"); // 销毁窗口
    }

    return image; // 返回图像对象
}


// 加载文件夹内所有图片
vector<ImageData> loadAllImages(const string &folder_path, bool show_img, bool ignore_errors)
{
    vector<ImageData> images;

    // 支持的图片扩展名集合（可扩展）
    const unordered_set<string> valid_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tiff", ".tif", ".webp", ".ppm"};

    try
    {
        // 遍历目录
        for (const auto &entry : fs::directory_iterator(folder_path))
        {
            // 跳过非普通文件
            if (!entry.is_regular_file())
                continue;

            // 获取文件路径和扩展名
            const string path_str = entry.path().string();
            string extension = entry.path().extension().string();

            // 统一转换为小写
            transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            // 检查扩展名有效性
            if (valid_extensions.count(extension) == 0)
                continue;

            try
            {
                // 加载图片并添加到结果列表
                Mat img = loadImage(path_str, show_img);
                images.push_back({move(img), entry.path().filename().string()});
            }
            catch (const exception &e)
            {
                if (!ignore_errors)
                    throw; // 根据参数决定是否抛出异常
                cerr << "[WARNING] Failed to load: " << path_str
                     << " - " << e.what() << endl;
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        throw runtime_error("Directory access error: " + string(e.what()));
    }

    // 添加结果检查
    if (images.empty())
    {
        throw runtime_error("No valid images found in: " + folder_path);
    }

    return images;
}


// 图像保存
void saveResults(const std::vector<ImageData> &data,
                const std::string &output_dir,
                const std::string &suffix)
{
    fs::create_directories(output_dir);

    for (const auto &item : data)
    {
        const std::string output_path =
            (fs::path(output_dir) / item.getOutputName(suffix)).string();

        if (!cv::imwrite(output_path, item.image))
        {
            throw std::runtime_error("Failed to save: " + output_path);
        }
    }
}
