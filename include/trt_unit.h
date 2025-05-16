#ifndef TRT_UNIT_H
#define TRT_UNIT_H

#include "trt_base.h"


static auto StreamDeleter = [](cudaStream_t* pStream) {
    if (pStream)
    {
        static_cast<void>(cudaStreamDestroy(*pStream));
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}


// 权重创建函数
nvinfer1::Weights make_weights(float* ptr, int n);
// 网络构建函数
UniqueNetwork buildNetwork(TRTLogger& logger, UniqueBuilder& builder, const char* onnx_path=nullptr);
UniqueNetwork buildNetwork_FC(TRTLogger& logger, UniqueBuilder& builder);
UniqueNetwork buildNetwork_CNN(TRTLogger& logger, UniqueBuilder& builder);
// 引擎构建函数
UniqueEngine buildEngine(UniqueBuilder& builder, UniqueNetwork& network, UniqueConfig& config, bool dynamic_Dim = false);
// 模型序列化保存函数
void saveEngine(UniqueEngine& engine, const char* filename);


#endif