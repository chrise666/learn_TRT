#ifndef TRT_UNIT_H
#define TRT_UNIT_H

#include "trt_base.h"

// 权重创建函数
nvinfer1::Weights make_weights(float* ptr, int n);
// 网络构建函数
UniqueNetwork buildNetwork(UniqueBuilder& builder, bool from_onnx=false, const char* onnx_path=nullptr);
// 引擎构建函数
UniqueEngine buildEngine(UniqueBuilder& builder, UniqueNetwork& network, UniqueConfig& config, bool dynamic_Dim = false);
// 模型序列化保存函数
void saveEngine(UniqueEngine& engine, const char* filename);

#endif