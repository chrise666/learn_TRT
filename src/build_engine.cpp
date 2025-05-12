#include "trt_unit.h"
#include "build_model.h"

using namespace nvinfer1;


void createEngine(TRTLogger logger) {
    // 1. 创建基础组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    UniqueBuilder builder(createInferBuilder(logger));
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    UniqueConfig config(builder->createBuilderConfig());
    
    // 2. 构建网络
    UniqueNetwork network = buildNetwork_FC(buildNetwork(logger, builder));

    // 3. 构建引擎
    UniqueEngine engine = buildEngine(builder, network, config);

    // 4. 保存模型
    saveEngine(engine, "trtmodel.engine");

    printf("Done.\n");
}

// void createEnginefromONNX(const char* onnx_path, const char* engine_name) {
//     // 1. 创建基础组件
//     // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
//     nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
//     // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
//     nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
//     // 2. 构建网络
//     const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//     nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
//     // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
//     nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
//     if(!parser->parseFromFile(onnx_path, 1)){
//         printf("Failed to parser demo.onnx\n");
//         cleanup(builder, config, nullptr, nullptr);
//     }

//     // 3. 构建引擎
//     nvinfer1::ICudaEngine* engine = buildEngine(builder, network, config, true);
//     if (!engine) {
//         printf("Build engine failed.\n");
//         cleanup(builder, config, network, nullptr);
//     }

//     // 4. 保存模型
//     saveEngine(engine, engine_name);

//     // 5. 清理资源
//     parser->destroy();
//     cleanup(builder, config, network, engine);

//     printf("Done.\n");
// }
