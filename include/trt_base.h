#ifndef TRT_BASE_H
#define TRT_BASE_H

// tensorRT include
#include <NvInfer.h>            // 编译用的头文件
#include <NvInferRuntime.h>     // 推理用的运行时头文件
#include <NvOnnxParser.h>       // onnx解析器的头文件

// cuda include
#include <cuda_runtime.h>

// opencv include
#include <opencv2/opencv.hpp>

// c++ standard library
#include <memory>
#include <iostream>


using Severity = nvinfer1::ILogger::Severity;

// 定义常用参数
struct Params
{
    bool dynamic_Dim{false};    //!< Allow running the network with dynamic dimensions.
    bool int8{false};           //!< Allow runnning the network in Int8 mode.
    bool fp16{false};           //!< Allow running the network in FP16 mode.
    bool bf16{false};           //!< Allow running the network in BF16 mode.
    const char* onnx_path;      //!< Filename of ONNX file of a network
    const char* engine_path;    //!< Filename of the serialized engine
};

inline const char* severity_string(Severity t){
    switch(t){
        case Severity::kINTERNAL_ERROR: return "internal_error";
        case Severity::kERROR:   return "error";
        case Severity::kWARNING: return "warning";
        case Severity::kINFO:    return "info";
        case Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

//!
//! \class TRTLogger
//!
//! \brief 管理TensorRT日志的类
//!
//! \details 此类为TensorRT对象提供了一个通用接口，用于将信息记录到控制台。
//!
//! 打印带颜色的字符，格式如下：
//! printf("\033[47;33m打印的文本\033[0m");
//! 其中 \033[ 是起始标记
//!      47    是背景颜色
//!      ;     分隔符
//!      33    文字颜色
//!      m     开始标记结束
//!      \033[0m 是终止标记
//! 其中背景颜色或者文字颜色可不写
//! 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
//!
class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){

            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

// 智能指针类型别名
using UniqueBuilder = std::unique_ptr<nvinfer1::IBuilder>;
using UniqueConfig = std::unique_ptr<nvinfer1::IBuilderConfig>;
using UniqueNetwork = std::unique_ptr<nvinfer1::INetworkDefinition>;
using UniqueParser = std::unique_ptr<nvonnxparser::IParser>;
using UniqueEngine = std::unique_ptr<nvinfer1::ICudaEngine>;
using UniqueRuntime = std::unique_ptr<nvinfer1::IRuntime>;
using UniqueExecutionContext = std::unique_ptr<nvinfer1::IExecutionContext>;
using UniqueHostMemory = std::unique_ptr<nvinfer1::IHostMemory>;

// INT8 校准器
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::vector<std::string>& imagePaths,
                          const std::string& calibrationCacheFile,
                          const nvinfer1::Dims& inputDims)
        : mCalibrationCacheFile(calibrationCacheFile), mInputDims(inputDims),
          mImageIndex(0), mImagePaths(imagePaths) {
        
        // 验证输入维度
        if (mInputDims.nbDims != 4 || mInputDims.d[1] != 3) {
            throw std::runtime_error("输入维度应为[N,C,H,W]且C=3");
        }
        
        // 分配 GPU 内存
        size_t inputSize = mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3] * sizeof(float);
        CUDA_CHECK(cudaMalloc(&mDeviceInput, inputSize));
    }

    ~Int8EntropyCalibrator() override {
        CUDA_CHECK(cudaFree(mDeviceInput));
    }

    // 获取批次大小
    int getBatchSize() const noexcept override {
        return mInputDims.d[0]; // 通常使用批次大小1进行校准
    }

    // 获取下一批校准数据
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mImageIndex >= mImagePaths.size()) {
            return false; // 所有图像已处理
        }

        // 加载并预处理图像
        cv::Mat image = cv::imread(mImagePaths[mImageIndex]);
        if (image.empty()) {
            std::cerr << "无法加载图像: " << mImagePaths[mImageIndex] << std::endl;
            return false;
        }

        // 预处理步骤
        cv::Mat processed;
        PreprocessImage(image, processed);

        // 将数据复制到 GPU
        CUDA_CHECK(cudaMemcpy(mDeviceInput, processed.data, 
                             processed.total() * processed.elemSize(),
                             cudaMemcpyHostToDevice));

        bindings[0] = mDeviceInput;
        mImageIndex++;
        return true;
    }

    // 读取校准缓存
    const void* readCalibrationCache(size_t& length) noexcept override {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationCacheFile, std::ios::binary);
        if (input.good()) {
            input.seekg(0, std::ios::end);
            length = input.tellg();
            input.seekg(0, std::ios::beg);
            mCalibrationCache.resize(length);
            input.read(mCalibrationCache.data(), length);
            return mCalibrationCache.data();
        }
        return nullptr;
    }

    // 写入校准缓存
    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        std::ofstream output(mCalibrationCacheFile, std::ios::binary);
        if (output.good()) {
            output.write(static_cast<const char*>(cache), length);
        }
    }

private:
    // 图像预处理函数
    void PreprocessImage(cv::Mat& input, cv::Mat& output) {
        // 1. 调整大小 (根据模型输入尺寸)
        cv::resize(input, input, cv::Size(mInputDims.d[3], mInputDims.d[2]));
        
        // 2. 转换数据类型并归一化
        input.convertTo(output, CV_32FC3, 1.0 / 255.0);
        
        // 3. 减去均值并缩放 (根据模型要求)
        cv::Scalar mean(0.485, 0.456, 0.406); // ImageNet均值
        cv::Scalar std(0.229, 0.224, 0.225);  // ImageNet标准差
        output = (output - mean) / std;
        
        // 4. 转换为CHW格式 (TensorRT要求)
        cv::dnn::blobFromImage(output, output);
    }

    static void CUDA_CHECK(cudaError_t code) {
        if (code != cudaSuccess) {
            throw std::runtime_error("CUDA错误: " + std::string(cudaGetErrorString(code)));
        }
    }

    std::vector<std::string> mImagePaths;
    size_t mImageIndex;
    void* mDeviceInput{nullptr};
    Dims mInputDims;
    std::string mCalibrationCacheFile;
    std::vector<char> mCalibrationCache;
};


#endif