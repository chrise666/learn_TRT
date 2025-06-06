#include "trt_unit.h"
#include <stdexcept>

using namespace nvinfer1;

// 权重创建函数
Weights make_weights(float* ptr, int n) {
    Weights w;
    w.count = n;
    w.type = DataType::kFLOAT;
    w.values = ptr;
    return w;
}

// 网络构建函数
UniqueNetwork buildNetwork(TRTLogger& logger, UniqueBuilder& builder, Params& mParams) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    auto network = UniqueNetwork(builder->createNetworkV2(explicitBatch));
    if (!network) {
        throw std::runtime_error("Failed to create network");
    }

    // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = UniqueParser(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile(mParams.onnx_path, 2)){
        printf("Error: %s\n", parser->getError(0)->desc());
        throw std::runtime_error("Failed to parse ONNX model");
    }

    return network;
}

// 引擎构建函数
UniqueEngine buildEngine(UniqueBuilder& builder, UniqueNetwork& network, UniqueConfig& config, Params& mParams) {
    // 设置工作空间内存池大小（替代 setMaxWorkspaceSize）
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 28); // 256MB
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);

    // 设置量化模式
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.bf16)
    {
        config->setFlag(BuilderFlag::kBF16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(nullptr);
    }

    // 设置动态尺寸
    if (mParams.dynamic_Dim) {
        auto input_tensor = network->getInput(0);
        int input_channel = input_tensor->getDimensions().d[1];
        int input_height = input_tensor->getDimensions().d[2];
        int input_width = input_tensor->getDimensions().d[3];

        // 如果模型有多个输入，则必须多个profile
        auto profile = builder->createOptimizationProfile();
        // 配置输入的最小、最优、最大的范围
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, input_height, input_width));
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, input_height, input_width));
        // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(10, input_channel, input_height, input_width));
        // 添加到配置
        config->addOptimizationProfile(profile);
    }

    //TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    auto engine = UniqueEngine(builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        throw std::runtime_error("Failed to build engine");
    }

    return engine;
}

// 模型序列化保存函数
void saveEngine(UniqueEngine& engine, Params& mParams) {
    auto model_data = UniqueHostMemory(engine->serialize());
    FILE* f = fopen(mParams.engine_path, "wb");
    if (!f) {
        throw std::runtime_error("Failed to open file");
    }
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
}