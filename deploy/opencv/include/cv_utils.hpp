#ifndef CV_UTILS_HPP
#define CV_UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

#ifdef WITH_CUDA
#include <opencv2/core/cuda.hpp>
#include <cuda_runtime.h>
#endif


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
    // 缩放模式枚举
    enum ResizeMode {
        NEAREST,        // 最近邻插值
        LINEAR,         // 双线性插值
        CUBIC,          // 双三次插值
        AREA,           // 区域插值（缩小图像时推荐）
        LANCZOS4,       // Lanczos插值
        AUTO            // 自动选择最佳方法
    };

    // 缩放策略枚举
    enum ResizeStrategy {
        EXACT_SIZE,     // 精确指定宽高
        SCALE_FACTOR,   // 按比例缩放
        MAX_DIMENSION,  // 限制最大尺寸
        MIN_DIMENSION,  // 限制最小尺寸
        FIT_AREA        // 适应指定区域
    };

    // 配置参数结构体
    struct Config {
        ResizeMode mode = LINEAR;
        ResizeStrategy strategy = EXACT_SIZE;
        cv::Size target_size;           // 用于EXACT_SIZE
        double scale_x = 1.0;           // 用于SCALE_FACTOR
        double scale_y = 1.0;           // 用于SCALE_FACTOR
        int max_dimension = 1024;       // 用于MAX_DIMENSION
        int min_dimension = 256;        // 用于MIN_DIMENSION
        int target_area = 1000000;      // 用于FIT_AREA
        bool keep_aspect_ratio = true;  // 是否保持宽高比
        bool use_gpu = false;           // 是否使用GPU加速
    };

    // 构造函数
    ImageResizer();
    
    // 单张图像缩放
    cv::Mat process(const cv::Mat &src, const Config &cfg) const;
    
    // 批量图像缩放
    std::vector<ImageData> batchProcess(const std::vector<ImageData> &images, const Config &cfg) const;

#ifdef WITH_CUDA
    // GPU加速版本
    cv::Mat processGPU(const cv::Mat &src, const Config &cfg) const;
    std::vector<ImageData> batchProcessGPU(const std::vector<ImageData> &images, const Config &cfg) const;
#endif

private:
    // 计算目标尺寸
    cv::Size calculateTargetSize(const cv::Mat &src, const Config &cfg) const;
    
    // 获取OpenCV插值标志
    int getInterpolationFlag(ResizeMode mode) const;
    
    // CUDA可用性标志
    bool cuda_available_;
    
    // 私有实现方法
    cv::Mat resizeImpl(const cv::Mat &src, const Config &cfg) const;
};


// 加载单张图像
cv::Mat loadImage(const std::string &img_path);

// 加载多张图像
std::vector<ImageData> loadAllImages(const std::string &folder_path);

// 保存图像
void saveResults(const std::vector<ImageData>& data,
                const std::string& output_dir,
                const std::string& suffix = "_modified");


#endif // CV_UTILS_HPP