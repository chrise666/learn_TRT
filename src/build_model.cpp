#include "trt_unit.h"


using namespace nvinfer1;


// 构建FC网络
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
void buildNetwork_FC(UniqueNetwork& network) {
    const int num_input = 3;   // in_channel
    const int num_output = 2;  // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb 
    float layer1_bias_values[]   = {0.3, 0.8};

    //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    ITensor* input = network->addInput("image", DataType::kFLOAT, Dims4{1, num_input, 1, 1});
    int32_t const batch = input->getDimensions().d[0];
    int32_t const mmInputs = input->getDimensions().d[1] * input->getDimensions().d[2] * input->getDimensions().d[3];
    auto inputReshape = network->addShuffle(*input);
    inputReshape->setReshapeDimensions(Dims{2, {batch, mmInputs}});

    // 1. 添加权重常量层
    Weights layer1_weight = make_weights(layer1_weight_values, num_input*num_output);
    Weights layer1_bias   = make_weights(layer1_bias_values, num_output);

    // 2. 创建权重张量
    IConstantLayer* weight_layer = network->addConstant(
        Dims{2, {num_output, mmInputs}}, // 注意维度顺序
        layer1_weight
    );

    // 3. 添加矩阵乘法操作：output = input * weight^T
    auto matrix_mult = network->addMatrixMultiply(
        *inputReshape->getOutput(0),             // input tensor [1,3,1,1]
        MatrixOperation::kNONE,        // 不转置输入
        *weight_layer->getOutput(0),             // weight tensor [3,2,1,1]
        MatrixOperation::kTRANSPOSE    // 转置权重矩阵
    );
    
    // 4. 添加偏置
    IConstantLayer* bias_layer = network->addConstant(
        Dims{2, {1, num_output}},          // [1, 2]
        layer1_bias
    );
    auto bias_add = network->addElementWise(
        *matrix_mult->getOutput(0),
        *bias_layer->getOutput(0),
        ElementWiseOperation::kSUM
    );

    // 5. 添加激活层
    auto prob = network->addActivation(*bias_add->getOutput(0), ActivationType::kSIGMOID);  // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    // 将我们需要的prob标记为输出    
    network->markOutput(*prob->getOutput(0));
}

// 构建CNN网络
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
// UniqueNetwork buildNetwork_CNN(UniqueNetwork network) {
//     const int num_input = 1;
//     const int num_output = 1;
//     float layer1_weight_values[] = {
//         1.0, 2.0, 3.1, 
//         0.1, 0.1, 0.1, 
//         0.2, 0.2, 0.2
//     }; // 行优先
//     float layer1_bias_values[]   = {0.0};

//     //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
//     ITensor* input = network->addInput("image", DataType::kFLOAT, Dims4(-1, num_input, -1, -1));
//     Weights layer1_weight = make_weights(layer1_weight_values, 9);
//     Weights layer1_bias   = make_weights(layer1_bias_values, 1);

//     auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
//     layer1->setPadding(nvinfer1::DimsHW(1, 1));

//     auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU); // *(layer1->getOutput(0))
     
//     // 将我们需要的prob标记为输出
//     network->markOutput(*prob->getOutput(0));

//     return network;
// }
