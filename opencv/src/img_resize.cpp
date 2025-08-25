#include "cv_utils.hpp"
#include <stdexcept>
#include <cmath>


using namespace cv;
using namespace std;


// 检查CUDA支持
bool ImageResizer::checkCudaSupport() {
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
}

// 计算缩放信息和填充区域
ImageResizer::ResizeInfo ImageResizer::calculateResizeInfo(
    const cv::Size& src_size, 
    const cv::Size& target_size) 
{
    ResizeInfo info;
    
    // 计算宽高比
    double src_ratio = static_cast<double>(src_size.width) / src_size.height;
    double target_ratio = static_cast<double>(target_size.width) / target_size.height;
    
    // 当目标尺寸大于原始尺寸时
    if (target_size.width >= src_size.width && target_size.height >= src_size.height) {
        info.scale = 1.0; // 不缩放
        info.scaled_size = src_size;
        
        // 计算居中位置
        int x = (target_size.width - src_size.width) / 2;
        int y = (target_size.height - src_size.height) / 2;
        info.roi = cv::Rect(x, y, src_size.width, src_size.height);
    }
    // 当目标尺寸小于原始尺寸时
    else {
        // 计算缩放比例 (保持宽高比)
        double scale_w = static_cast<double>(target_size.width) / src_size.width;
        double scale_h = static_cast<double>(target_size.height) / src_size.height;
        info.scale = std::min(scale_w, scale_h);
        
        // 计算缩放后的尺寸
        info.scaled_size = cv::Size(
            static_cast<int>(std::round(src_size.width * info.scale)),
            static_cast<int>(std::round(src_size.height * info.scale))
        );
        
        // 计算居中位置
        int x = (target_size.width - info.scaled_size.width) / 2;
        int y = (target_size.height - info.scaled_size.height) / 2;
        info.roi = cv::Rect(x, y, info.scaled_size.width, info.scaled_size.height);
    }
    
    return info;
}

// CPU实现
Mat ImageResizer::smartResizeCPU(const Mat& src, 
                               const Size& target_size,
                               const Scalar& bg_color) 
{
    if (src.empty()) throw invalid_argument("Input image is empty");
    
    // 创建目标图像并填充背景色
    Mat dst(target_size, src.type(), bg_color);
    
    // 计算缩放和填充信息
    ResizeInfo info = calculateResizeInfo(src.size(), target_size);
    
    // 当需要缩放时
    if (info.scale != 1.0) {
        Mat resized;
        cv::resize(src, resized, info.scaled_size, 0, 0, INTER_LINEAR);
        resized.copyTo(dst(info.roi));
    } 
    // 不需要缩放时
    else {
        src.copyTo(dst(info.roi));
    }
    
    return dst;
}

// GPU实现
Mat ImageResizer::smartResizeGPU(const Mat& src, 
                               const Size& target_size,
                               const Scalar& bg_color) 
{
    if (src.empty()) throw invalid_argument("Input image is empty");
    if (!checkCudaSupport()) throw runtime_error("CUDA not available");
    
    // 创建GPU内存对象
    cuda::GpuMat d_src, d_resized, d_dst;
    
    // 上传原始图像到GPU
    d_src.upload(src);
    
    // 创建目标图像并填充背景色
    d_dst.create(target_size, src.type());
    d_dst.setTo(bg_color);
    
    // 计算缩放和填充信息
    ResizeInfo info = calculateResizeInfo(src.size(), target_size);
    
    // 当需要缩放时
    if (info.scale != 1.0) {
        // GPU缩放
        // cuda::resize(d_src, d_resized, info.scaled_size, 0, 0, INTER_LINEAR);
        // 复制到目标位置
        cuda::GpuMat d_roi(d_dst, info.roi);
        d_resized.copyTo(d_roi);
    } 
    // 不需要缩放时
    else {
        cuda::GpuMat d_roi(d_dst, info.roi);
        d_src.copyTo(d_roi);
    }
    
    // 下载结果到CPU
    Mat dst;
    d_dst.download(dst);
    
    return dst;
}

// 统一接口
Mat ImageResizer::smartResize(const Mat& src,
                            const Size& target_size,
                            Mode mode,
                            int gpu_threshold,
                            const Scalar& bg_color) 
{
    // 参数验证
    if (src.empty()) throw invalid_argument("Input image is empty");
    if (target_size.width <= 0 || target_size.height <= 0)
        throw invalid_argument("Invalid target size");
    
    // 模式自动选择逻辑
    bool use_gpu = false;
    switch (mode) {
        case CPU_ONLY:
            use_gpu = false;
            break;
        case GPU_ONLY:
            if (!checkCudaSupport()) 
                throw runtime_error("GPU mode requested but not available");
            use_gpu = true;
            break;
        case AUTO:
        default:
            use_gpu = checkCudaSupport() && 
                     (src.total() > gpu_threshold);
            break;
    }
    
    return use_gpu ? smartResizeGPU(src, target_size, bg_color) 
                  : smartResizeCPU(src, target_size, bg_color);
}
