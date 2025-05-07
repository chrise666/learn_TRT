#include "trt_base.h"
#include <stdio.h>
#include <memory>
#include <stdexcept>


// 自定义删除器用于智能指针管理TensorRT对象
struct TensorRTDeleter {
    void operator()(nvinfer1::IHostMemory* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::IBuilder* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::IBuilderConfig* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::INetworkDefinition* ptr) { if (ptr) delete ptr; }
    void operator()(nvinfer1::ICudaEngine* ptr) { if (ptr) delete ptr; }
};
// 智能指针类型别名
using UniqueBuilder = std::unique_ptr<nvinfer1::IBuilder, TensorRTDeleter>;
using UniqueConfig = std::unique_ptr<nvinfer1::IBuilderConfig, TensorRTDeleter>;
using UniqueNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, TensorRTDeleter>;
using UniqueEngine = std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter>;
using UniqueHostMemory = std::unique_ptr<nvinfer1::IHostMemory, TensorRTDeleter>;


// 权重创建函数
nvinfer1::Weights make_weights(float* ptr, int n) {
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

// 网络构建函数
/*
    Network definition:

    image
      |
    linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
      |
    sigmoid
      |
    prob
*/
UniqueNetwork buildNetwork(nvinfer1::IBuilder* builder) {
    const int num_input = 3;   // in_channel
    const int num_output = 2;  // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb 
    float layer1_bias_values[]   = {0.3, 0.8};

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    UniqueNetwork network(builder->createNetworkV2(explicitBatch));

    //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    nvinfer1::ITensor* input = network->addInput("image", 
                                                nvinfer1::DataType::kFLOAT, 
                                                nvinfer1::Dims4{1, num_input, 1, 1});
    int32_t const batch = input->getDimensions().d[0];
    int32_t const mmInputs = input->getDimensions().d[1] * input->getDimensions().d[2] * input->getDimensions().d[3];
    auto inputReshape = network->addShuffle(*input);
    inputReshape->setReshapeDimensions(nvinfer1::Dims{2, {batch, mmInputs}});

    // 1. 添加权重常量层
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, num_input*num_output);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, num_output);

    // 2. 创建权重张量
    nvinfer1::IConstantLayer* weight_layer = network->addConstant(
        nvinfer1::Dims{2, {num_output, mmInputs}}, // 注意维度顺序
        layer1_weight
    );

    // 3. 添加矩阵乘法操作：output = input * weight^T
    auto matrix_mult = network->addMatrixMultiply(
        *inputReshape->getOutput(0),             // input tensor [1,3,1,1]
        nvinfer1::MatrixOperation::kNONE,        // 不转置输入
        *weight_layer->getOutput(0),             // weight tensor [3,2,1,1]
        nvinfer1::MatrixOperation::kTRANSPOSE    // 转置权重矩阵
    );
    
    // 4. 添加偏置
    nvinfer1::IConstantLayer* bias_layer = network->addConstant(
        nvinfer1::Dims{2, {1, num_output}},          // [1, 2]
        layer1_bias
    );
    auto bias_add = network->addElementWise(
        *matrix_mult->getOutput(0),
        *bias_layer->getOutput(0),
        nvinfer1::ElementWiseOperation::kSUM
    );

    // 5. 添加激活层
    auto prob = network->addActivation(*bias_add->getOutput(0), nvinfer1::ActivationType::kSIGMOID);  // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    // 将我们需要的prob标记为输出    
    network->markOutput(*prob->getOutput(0));

    return network;
}

// 引擎构建函数
UniqueEngine buildEngine(nvinfer1::IBuilder* builder, UniqueNetwork& network, nvinfer1::IBuilderConfig* config) {
    // 设置工作空间内存池大小（替代 setMaxWorkspaceSize）
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28); // 256MB
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); 

    //TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        throw std::runtime_error("Failed to build engine");
    }
    return UniqueEngine(engine);
}

// 模型序列化保存函数
void saveEngine(nvinfer1::ICudaEngine* engine, const char* filename) {
    UniqueHostMemory model_data(engine->serialize());
    FILE* f = fopen(filename, "wb");
    if (!f) {
        throw std::runtime_error("Failed to open file");
    }
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
}


void createEngine() {
    TRTLogger logger; // logger是必要的，用来捕捉warning和info等

    // 1. 创建基础组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    UniqueBuilder builder(createInferBuilder(logger));
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    UniqueConfig config(builder->createBuilderConfig());
    
    // 2. 构建网络
    UniqueNetwork network = buildNetwork(builder.get());
    if (!network) {
        printf("Build network failed.\n");
    }

    // 3. 构建引擎
    UniqueEngine engine = buildEngine(builder.get(), network, config.get());
    if (!engine) {
        printf("Build engine failed.\n");
    }

    // 4. 保存模型
    saveEngine(engine.get(), "engine.trtmodel");

    printf("Done.\n");
}