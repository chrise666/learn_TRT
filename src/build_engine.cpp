#include "trt_unit.h"

using namespace nvinfer1;


bool createEngine(TRTLogger& logger, const char* onnx_path, bool dynamic_Dim) {
    // 1. 创建基础组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    auto builder = UniqueBuilder(createInferBuilder(logger));
    if (!builder)
    {
        return false;
    }
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    auto config = UniqueConfig(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    
    // 2. 构建网络
    // auto network = buildNetwork(logger, builder, onnx_path);
    // auto network = buildNetwork_CNN(logger, builder);
    auto network = buildNetwork_FC(logger, builder);
    if (!network)
    {
        return false;
    }

    // 3. 构建引擎
    // CUDA stream used for profiling by the builder.
    auto profileStream = makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    auto engine = buildEngine(builder, network, config, dynamic_Dim);
    if (!engine)
    {
        return false;
    }

    // 4. 保存模型
    saveEngine(engine, "trtmodel.engine");

    printf("Done.\n");
    
    return true;
}