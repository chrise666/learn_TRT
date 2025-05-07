#ifndef TRT_BASE_H
#define TRT_BASE_H

// tensorRT include
#include <NvInfer.h>            // 编译用的头文件
#include <NvInferRuntime.h>     // 推理用的运行时头文件
#include <NvOnnxParser.h>       // onnx解析器的头文件

// cuda include
#include <cuda_runtime.h>


using namespace nvinfer1;


inline const char* severity_string(ILogger::Severity t){
    switch(t){
        case ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case ILogger::Severity::kERROR:   return "error";
        case ILogger::Severity::kWARNING: return "warning";
        case ILogger::Severity::kINFO:    return "info";
        case ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public ILogger{
public:
    virtual void log(Severity severity, AsciiChar const* msg) noexcept override{
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


// void createEngine();
// void createEnginefromONNX(const char* onnx_path, const char* engine_name);

// void inference();
// void inference_dynamic(const char* engine_path);

#endif