#include "cv_utils.hpp"


using namespace cv;
using namespace std;


ImageResizer::ImageResizer() {
#ifdef WITH_CUDA
    // 检查CUDA可用性
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    cuda_available_ = (err == cudaSuccess) && (device_count > 0);
    
    if (cuda_available_) {
        std::cout << "CUDA加速可用，设备数量: " << device_count << std::endl;
    } else {
        std::cout << "CUDA加速不可用，使用CPU处理" << std::endl;
    }
#else
    cuda_available_ = false;
#endif
}

/**
 * @brief 执行单张图像缩放
 * @param src 输入图像
 * @param cfg 缩放配置
 * @return 缩放后的图像
 * @throws std::invalid_argument 无效参数
 */
cv::Mat ImageResizer::process(const cv::Mat &src, const Config &cfg) const
{
    if (src.empty()) {
        throw std::invalid_argument("输入图像为空");
    }
    
    // 根据配置选择处理方式
    if (cfg.use_gpu && cuda_available_) {
#ifdef WITH_CUDA
        return resizeGPU(src, cfg);
#else
        std::cerr << "警告: 编译时未启用CUDA支持，使用CPU处理" << std::endl;
#endif
    }
    
    return resizeImpl(src, cfg);
}

/**
 * @brief 批量处理图像
 * @param images 输入图像集合
 * @param cfg 缩放配置
 * @return 处理后的图像集合
 */
vector<ImageData> ImageResizer::batchProcess(const vector<ImageData> &images, const Config &cfg) const
{
    std::vector<ImageData> results;
    results.reserve(images.size());
    
    for (const auto &img : images)
    {
        try
        {
            Mat processed = process(img.image, cfg);
            results.push_back({processed, img.filename});
        }
        catch (const std::exception& e)
        {
            std::cerr << "Failed to process " << img.filename
                      << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

/**
 * @brief 图像缩放函数
 * @param src 输入图像
 * @param cfg 缩放配置
 * @return 缩放后的图像
 */
cv::Mat ImageResizer::resizeImpl(const cv::Mat &src, const Config &cfg) const
{
    // 计算目标尺寸
    cv::Size target_size = calculateTargetSize(src, cfg);
    
    if (target_size == src.size()) {
        return src.clone();  // 尺寸相同，直接返回副本
    }
    
    // 获取插值方法
    int interpolation = getInterpolationFlag(cfg.mode);
    
    // 执行缩放
    cv::Mat dst;
    cv::resize(src, dst, target_size, 0, 0, interpolation);
    
    return dst;
}

cv::Size ImageResizer::calculateTargetSize(const cv::Mat &src, const Config &cfg) const
{
    int src_w = src.cols;
    int src_h = src.rows;
    double aspect_ratio = static_cast<double>(src_w) / src_h;
    
    switch (cfg.strategy) {
        case EXACT_SIZE:
            return cfg.target_size;
            
        case SCALE_FACTOR:
            if (cfg.keep_aspect_ratio) {
                // 保持宽高比，使用相同的缩放因子
                double scale = (cfg.scale_x + cfg.scale_y) / 2.0;
                return cv::Size(
                    static_cast<int>(src_w * scale),
                    static_cast<int>(src_h * scale)
                );
            } else {
                return cv::Size(
                    static_cast<int>(src_w * cfg.scale_x),
                    static_cast<int>(src_h * cfg.scale_y)
                );
            }
            
        case MAX_DIMENSION:
            if (cfg.keep_aspect_ratio) {
                if (src_w > src_h) {
                    return cv::Size(
                        cfg.max_dimension,
                        static_cast<int>(cfg.max_dimension / aspect_ratio)
                    );
                } else {
                    return cv::Size(
                        static_cast<int>(cfg.max_dimension * aspect_ratio),
                        cfg.max_dimension
                    );
                }
            } else {
                return cv::Size(cfg.max_dimension, cfg.max_dimension);
            }
            
        case MIN_DIMENSION:
            if (cfg.keep_aspect_ratio) {
                if (src_w < src_h) {
                    return cv::Size(
                        cfg.min_dimension,
                        static_cast<int>(cfg.min_dimension / aspect_ratio)
                    );
                } else {
                    return cv::Size(
                        static_cast<int>(cfg.min_dimension * aspect_ratio),
                        cfg.min_dimension
                    );
                }
            } else {
                return cv::Size(cfg.min_dimension, cfg.min_dimension);
            }
            
        case FIT_AREA:
            if (cfg.keep_aspect_ratio) {
                // 计算适应目标区域的最大尺寸
                double scale = sqrt(static_cast<double>(cfg.target_area) / (src_w * src_h));
                return cv::Size(
                    static_cast<int>(src_w * scale),
                    static_cast<int>(src_h * scale)
                );
            } else {
                // 非保持宽高比，尝试找到最接近的整数尺寸
                int w = static_cast<int>(sqrt(cfg.target_area * aspect_ratio));
                int h = static_cast<int>(cfg.target_area / w);
                return cv::Size(w, h);
            }
            
        default:
            throw std::invalid_argument("不支持的缩放策略");
    }
}

int ImageResizer::getInterpolationFlag(ResizeMode mode) const
{
    switch (mode) {
        case NEAREST: return INTER_NEAREST;
        case LINEAR: return INTER_LINEAR;
        case CUBIC: return INTER_CUBIC;
        case AREA: return INTER_AREA;
        case LANCZOS4: return INTER_LANCZOS4;
        case AUTO:
            // 自动选择最佳插值方法
            // 缩小图像使用AREA，放大图像使用LANCZOS4
            // 这里需要知道原始和目标尺寸，但在方法选择时不知道
            return INTER_LINEAR; // 默认使用线性
        default: return INTER_LINEAR;
    }
}

#ifdef WITH_CUDA
/**
 * @brief 执行GPU上单张图像缩放
 * @param src 输入图像
 * @param cfg 缩放配置
 * @return 缩放后的图像
 */
cv::Mat ImageResizer::processGPU(const cv::Mat &src, const Config &cfg) const
{
    // 计算目标尺寸
    cv::Size target_size = calculateTargetSize(src, cfg);
    
    if (target_size == src.size()) {
        return src.clone();  // 尺寸相同，直接返回副本
    }
    
    // 上传到GPU
    cv::cuda::GpuMat d_src, d_dst;
    d_src.upload(src);
    
    // 获取插值方法
    int interpolation = getInterpolationFlag(cfg.mode);
    
    // GPU缩放
    cv::cuda::resize(d_src, d_dst, target_size, 0, 0, interpolation);
    
    // 下载到CPU
    cv::Mat dst;
    d_dst.download(dst);
    
    return dst;
}

/**
 * @brief GPU上批量处理图像
 * @param images 输入图像集合
 * @param cfg 缩放配置
 * @return 处理后的图像集合
 */
vector<ImageData> ImageResizer::batchProcessGPU(const vector<ImageData> &images, const Config &cfg) const
{
    std::vector<ImageData> results;
    results.reserve(images.size());
    
    for (const auto &img : images)
    {
        try
        {
            Mat processedGPU = processGPU(img.image, cfg);
            results.push_back({processedGPU, img.filename});
        }
        catch (const std::exception& e)
        {
            std::cerr << "Failed to process " << img.filename
                      << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}
#endif
