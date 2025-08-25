#include "cv_utils.hpp"

using namespace cv;
using namespace std;

/**
 * @brief 执行单张图像裁剪
 * @param src 输入图像
 * @param cfg 裁剪配置
 * @return 裁剪后的图像
 * @throws std::invalid_argument 无效参数
 * @throws std::runtime_error 处理失败
 */
cv::Mat ImageCropper::process(const cv::Mat &src, const Config &cfg) const
{
    if (src.empty())
    {
        throw std::invalid_argument("Input image is empty");
    }
    validateConfig(cfg); // 参数验证

    switch (cfg.mode)
    {
    case FIXED_ROI:
        return cropImage(src, cfg.roi, cfg.auto_adjust);
    case CENTER_CROP:
        if (cfg.target_size.empty())
        {
            throw std::invalid_argument("Target size not specified for center crop");
        }
        return centerCrop(src, cfg.target_size);
    case ASPECT_RATIO_CROP:
        return aspectRatioCrop(src, cfg.aspect_ratio, cfg.max_size);
    case NO_CROP:
    default:
        return src.clone();
    }
}

/**
 * @brief 批量处理图像
 * @param images 输入图像集合
 * @param cfg 裁剪配置
 * @return 处理后的图像集合
 */
vector<ImageData> ImageCropper::batchProcess(const vector<ImageData> &images,
                                             const Config &cfg) const
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
        catch (const std::exception &e)
        {
            // 记录错误并跳过无效图像
            std::cerr << "Failed to process " << img.filename
                      << ": " << e.what() << std::endl;
        }
    }
    return results;
}

/**
 * @brief 通用图像裁剪函数
 * @param src 输入图像
 * @param roi 裁剪区域 (x, y, width, height)
 * @param auto_adjust 当ROI超出边界时自动调整区域
 * @return 裁剪后的图像
 * @throws runtime_error 输入图像无效或ROI完全超出图像范围
 */
cv::Mat ImageCropper::cropImage(const cv::Mat &src,
                                cv::Rect roi,
                                bool auto_adjust = true) const
{
    // 获取图像边界
    const int img_w = src.cols;
    const int img_h = src.rows;

    // 自动调整ROI
    if (auto_adjust)
    {
        roi.x = max(0, roi.x);
        roi.y = max(0, roi.y);
        roi.width = min(roi.width, img_w - roi.x);
        roi.height = min(roi.height, img_h - roi.y);
    }
    else
    {
        // 严格验证区域
        if (roi.x < 0 || roi.y < 0 ||
            roi.x + roi.width > img_w ||
            roi.y + roi.height > img_h)
        {
            throw runtime_error("ROI out of image boundaries");
        }
    }

    // 检查有效区域
    if (roi.width <= 0 || roi.height <= 0)
    {
        throw runtime_error("Invalid ROI dimensions");
    }

    // 执行裁剪
    return src(roi).clone();  // 使用clone保证数据独立
}

/**
 * @brief 居中裁剪函数
 * @param src 输入图像
 * @param size 目标尺寸 (width, height)
 * @return 居中裁剪后的图像
 */
cv::Mat ImageCropper::centerCrop(const cv::Mat &src,
                                 cv::Size size) const
{
    // 计算中心区域
    const int center_x = src.cols / 2;
    const int center_y = src.rows / 2;

    Rect roi(
        center_x - size.width / 2,  // x
        center_y - size.height / 2, // y
        size.width,                 // width
        size.height                 // height
    );

    return cropImage(src, roi);
}

/**
 * @brief 比例缩放裁剪（保持宽高比）
 * @param src 输入图像
 * @param aspect_ratio 宽高比 (width/height)
 * @param max_size 最大尺寸
 * @return 裁剪后的图像
 */
cv::Mat ImageCropper::aspectRatioCrop(const cv::Mat &src,
                                      float aspect_ratio,
                                      int max_size = 2048) const
{
    // 计算目标尺寸
    int target_w, target_h;
    if (src.cols * 1.0 / src.rows > aspect_ratio)
    {
        // 原图更宽，按高度计算
        target_h = min(src.rows, max_size);
        target_w = target_h * aspect_ratio;
    }
    else
    {
        // 原图更高，按宽度计算
        target_w = min(src.cols, max_size);
        target_h = target_w / aspect_ratio;
    }

    return centerCrop(src, Size(target_w, target_h));
}

// 配置参数验证
void ImageCropper::validateConfig(const Config &cfg) const
{
    if (cfg.mode == CENTER_CROP && cfg.target_size.empty())
    {
        throw std::invalid_argument("Center crop requires target size");
    }
    if (cfg.mode == ASPECT_RATIO_CROP && cfg.aspect_ratio <= 0)
    {
        throw std::invalid_argument("Invalid aspect ratio");
    }
}
