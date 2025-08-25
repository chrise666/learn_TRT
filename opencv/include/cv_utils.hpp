#ifndef CV_UTILS_HPP
#define CV_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>


struct ImageData
{
    cv::Mat image;
    std::string filename;

    // 实用方法：生成带后缀的文件名
    std::string getOutputName(const std::string& suffix = "_modified") const {
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


// 图像尺寸调整
class ImageResizer {
public:
    enum Mode { AUTO, CPU_ONLY, GPU_ONLY };
    
    /**
     * @brief 智能缩放函数
     * @param src 输入图像
     * @param target_size 目标尺寸
     * @param mode 处理模式
     * @param gpu_threshold GPU加速阈值
     * @param bg_color 背景填充颜色 (BGR格式)
     * @return 处理后的图像
     */
    static cv::Mat smartResize(const cv::Mat& src,
                                const cv::Size& target_size,
                                Mode mode = CPU_ONLY,
                                int gpu_threshold = 1024*768,
                                const cv::Scalar& bg_color = cv::Scalar(0, 0, 0));
    
    /**
     * @brief 获取缩放比例和填充区域
     * @param src_size 原始尺寸
     * @param target_size 目标尺寸
     * @return 包含缩放比例和填充信息的结构体
     */
    struct ResizeInfo {
        double scale;       // 缩放比例
        cv::Rect roi;       // 有效区域
        cv::Size scaled_size; // 缩放后的尺寸
    };
    
    static ResizeInfo calculateResizeInfo(const cv::Size& src_size, 
                                        const cv::Size& target_size);

private:
    // CPU实现
    static cv::Mat smartResizeCPU(const cv::Mat& src, 
                                const cv::Size& target_size,
                                const cv::Scalar& bg_color);
    
    // GPU实现
    static cv::Mat smartResizeGPU(const cv::Mat& src, 
                                const cv::Size& target_size,
                                const cv::Scalar& bg_color);
    
    // 检查GPU可用性
    static bool checkCudaSupport();
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


#endif // CV_UTILS_HPP
