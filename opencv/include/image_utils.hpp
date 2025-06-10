#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>


struct ImageData
{
    cv::Mat image;
    std::string filename;

    // 实用方法：生成带后缀的文件名
    std::string getOutputName(const std::string& suffix = "_cropped") const {
        size_t dotPos = filename.find_last_of(".");
        if(dotPos == std::string::npos) {
            return filename + suffix;
        }
        return filename.substr(0, dotPos) + suffix + filename.substr(dotPos);
    }
};


// 图像裁剪
class ImageCropper {
public:
    // 裁剪模式枚举
    enum CropMode {
        NO_CROP,         // 不裁剪
        FIXED_ROI,       // 固定区域裁剪
        CENTER_CROP,     // 居中裁剪
        ASPECT_RATIO_CROP// 比例裁剪
    };

    // 配置参数结构体
    struct Config {
        CropMode mode = NO_CROP;   // 裁剪模式
        cv::Rect roi;              // 用于FIXED_ROI模式
        cv::Size target_size;      // 用于CENTER_CROP模式
        float aspect_ratio = 1.0f; // 用于ASPECT_RATIO_CROP模式
        int max_size = 2048;       // 用于ASPECT_RATIO_CROP模式
        bool auto_adjust = true;   // 自动调整ROI边界

        // 构造函数便于初始化
        Config() = default;
        Config(CropMode m, cv::Rect r, bool adjust) 
            : mode(m), roi(r), auto_adjust(adjust) {}
        Config(CropMode m, cv::Size s) 
            : mode(m), target_size(s) {}
        Config(CropMode m, float ratio, int max) 
            : mode(m), aspect_ratio(ratio), max_size(max) {}
    };

    cv::Mat process(const cv::Mat& src, const Config& cfg) const;
    std::vector<ImageData> batchProcess(const std::vector<ImageData>& images, 
                                        const Config& cfg) const;

private:
    cv::Mat cropImage(const cv::Mat& src, cv::Rect roi, bool auto_adjust) const;
    cv::Mat centerCrop(const cv::Mat& src, cv::Size size) const;
    cv::Mat aspectRatioCrop(const cv::Mat& src, float aspect_ratio, int max_size) const;
    void validateConfig(const Config& cfg) const;
};


// 加载单张图像并显示（可选）
cv::Mat loadImage(const std::string &img_path, bool show_img = false);

// 加载多张图像并显示（可选）
std::vector<ImageData> loadAllImages(const std::string &folder_path, 
                                    bool show_img = false, 
                                    bool ignore_errors = true);

// 保存图像
void saveResults(const std::vector<ImageData>& data,
                const std::string& output_dir,
                const std::string& suffix = "_cropped");


#endif // IMAGE_UTILS_H
