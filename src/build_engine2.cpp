#include "trt_base.h"


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
nvinfer1::INetworkDefinition* buildNetwork_FC(nvinfer1::IBuilder* builder) {
    const int num_input = 3;   // in_channel
    const int num_output = 2;  // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb 
    float layer1_bias_values[]   = {0.3, 0.8};

    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);

    // 添加全连接层
    auto layer1 = network->addFullyConnected(
        *input, // 注意对input进行了解引用
        num_output, 
        layer1_weight, 
        layer1_bias
    );
    
    // 添加激活层
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);  // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    // 将我们需要的prob标记为输出    
    network->markOutput(*prob->getOutput(0));

    return network;
}

// 构建一个模型
/*
    Network definition:

    image
        |
    conv(3x3, pad=1)  input = 1, output = 1, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5], [0.2, 0.2, 0.1]], b=0.0
        |
    relu
        |
    prob
*/
nvinfer1::INetworkDefinition* buildNetwork_CNN(nvinfer1::IBuilder* builder) {
    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1, 
        0.1, 0.1, 0.1, 
        0.2, 0.2, 0.2
    }; // 行优先
    float layer1_bias_values[]   = {0.0};

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
    // 如果要使用动态shape，必须让NetworkDefinition的维度定义为-1，in_channel是固定的
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 1);
    auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
    layer1->setPadding(nvinfer1::DimsHW(1, 1));

    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU); // *(layer1->getOutput(0))
     
    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    return network;
}

// 引擎构建函数
nvinfer1::ICudaEngine* buildEngine(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, nvinfer1::IBuilderConfig* config, bool dynamic = false) {
    config->setMaxWorkspaceSize(1 << 28);
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib

    if (dynamic) {
        // 如果模型有多个输入，则必须多个profile
        auto profile = builder->createOptimizationProfile();
        auto input_tensor = network->getInput(0);
        int input_channel = input_tensor->getDimensions().d[1];
        int input_height = input_tensor->getDimensions().d[2];
        int input_width = input_tensor->getDimensions().d[3];

        // 配置输入的最小、最优、最大的范围
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, input_height, input_width));
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, input_height, input_width));
        // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
        profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(10, input_channel, input_height, input_width));
        // 添加到配置
        config->addOptimizationProfile(profile);
    }else{
        // 固定batch size
        builder->setMaxBatchSize(1); // 推理时 batchSize = 1
    }
    //TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    return builder->buildEngineWithConfig(*network, *config);
}

// 模型序列化保存函数
void saveEngine(nvinfer1::ICudaEngine* engine, const char* filename) {
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen(filename, "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
    model_data->destroy();
}

// 资源清理函数
void cleanup(nvinfer1::IBuilder* builder, 
            nvinfer1::IBuilderConfig* config, 
            nvinfer1::INetworkDefinition* network, 
            nvinfer1::ICudaEngine* engine) {
    if (engine) engine->destroy();
    if (network) network->destroy();
    if (config) config->destroy();
    if (builder) builder->destroy();
}

void createEngine() {
    TRTLogger logger; // logger是必要的，用来捕捉warning和info等

    // 1. 创建基础组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    // 2. 构建网络
    nvinfer1::INetworkDefinition* network = buildNetwork_CNN(builder);
    if (!network) {
        printf("Build network failed.\n");
        cleanup(builder, config, nullptr, nullptr);
    }

    // 3. 构建引擎
    nvinfer1::ICudaEngine* engine = buildEngine(builder, network, config, true);
    if (!engine) {
        printf("Build engine failed.\n");
        cleanup(builder, config, network, nullptr);
    }

    // 4. 保存模型
    saveEngine(engine, "engine.trtmodel");

    // 5. 清理资源
    cleanup(builder, config, network, engine);

    printf("Done.\n");
}

void createEnginefromONNX(const char* onnx_path, const char* engine_name) {
    TRTLogger logger; // logger是必要的，用来捕捉warning和info等

    // 1. 创建基础组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    
    // 2. 构建网络
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile(onnx_path, 1)){
        printf("Failed to parser demo.onnx\n");
        cleanup(builder, config, nullptr, nullptr);
    }

    // 3. 构建引擎
    nvinfer1::ICudaEngine* engine = buildEngine(builder, network, config, true);
    if (!engine) {
        printf("Build engine failed.\n");
        cleanup(builder, config, network, nullptr);
    }

    // 4. 保存模型
    saveEngine(engine, engine_name);

    // 5. 清理资源
    parser->destroy();
    cleanup(builder, config, network, engine);

    printf("Done.\n");
}
