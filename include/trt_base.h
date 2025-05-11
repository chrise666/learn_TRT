#ifndef TRT_BASE_H
#define TRT_BASE_H

// tensorRT include
#include <NvInfer.h>            // 编译用的头文件
#include <NvInferRuntime.h>     // 推理用的运行时头文件
#include <NvOnnxParser.h>       // onnx解析器的头文件

// cuda include
#include <cuda_runtime.h>

// c++ standard library
#include <memory>
#include <stdexcept>


using Severity = nvinfer1::ILogger::Severity;


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

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
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
extern TRTLogger logger;

// 自定义删除器用于智能指针管理TensorRT对象
struct TensorRTDeleter {
    void operator()(nvinfer1::IBuilder* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::IBuilderConfig* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::INetworkDefinition* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::ICudaEngine* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::IHostMemory* ptr) { if (ptr) delete ptr; }
    void operator()(nvonnxparser::IParser* ptr) { if (ptr) delete ptr; }
};

// 智能指针类型别名
using UniqueBuilder = std::unique_ptr<nvinfer1::IBuilder, TensorRTDeleter>;
using UniqueConfig = std::unique_ptr<nvinfer1::IBuilderConfig, TensorRTDeleter>;
using UniqueNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, TensorRTDeleter>;
using UniqueEngine = std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter>;
using UniqueHostMemory = std::unique_ptr<nvinfer1::IHostMemory, TensorRTDeleter>;
using UniqueParser = std::unique_ptr<nvonnxparser::IParser, TensorRTDeleter>;

#endif