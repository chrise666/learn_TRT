#include "trt_unit.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;


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
// float input_data_host[] = {1, 2, 3};
// float output_data_host[2];

// 读取数据并分配内存
float get_data() {

}

// 定义一个函数来加载文件
auto load_file(const std::string& file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) throw std::runtime_error("打开文件失败，请检查文件路径");
  
  in.seekg(0, std::ios::end);
  size_t length = in.tellg();
  
  std::vector<char> data(length);
  in.seekg(0, std::ios::beg);
  in.read(data.data(), length);
  in.close();
  
  return data;
}

bool inference(TRTLogger logger, const char* engine_path, float input_data_host[], float output_data_host[]) {
    // ------------------------------ 1. 准备模型并加载   ----------------------------    
    auto engine_data = load_file(engine_path);
    if (engine_data.empty())
    {
        return false;
    }

    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    auto runtime = UniqueRuntime(createInferRuntime(logger));
    if (!runtime)
    {
        throw std::runtime_error("Create infer runtime failed.");
        return false;
    }

    // 将模型读取到engine_data中，则可以对其进行反序列化以获得engine
    auto engine = UniqueEngine(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine)
    {
        throw std::runtime_error("Deserialize cuda engine failed.");
        return false;
    }

    auto execution_context = UniqueExecutionContext(engine->createExecutionContext());
    if (!execution_context)
    {
        throw std::runtime_error("Create execution context failed.");
        return false;
    }

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
   
    float* input_data_device = nullptr;
    float* output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    for (int32_t i = 0; i < engine->getNbIOTensors(); i++)
    {
      const char* name = engine->getIOTensorName(i);
      TensorIOMode mode = engine->getTensorIOMode(name);

      // 设置输入输出张量地址      
      if (mode == TensorIOMode::kINPUT) {
          std::cout << "输入张量: " << name << std::endl;
          execution_context->setInputTensorAddress(name, input_data_device);
      } else if (mode == TensorIOMode::kOUTPUT) {
          std::cout << "输出张量: " << name << std::endl;
          execution_context->setOutputTensorAddress(name, output_data_device);
      }
    }

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    bool success = execution_context->enqueueV3(stream);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
}

// void inference_dynamic(const char* engine_path){
//     // ------------------------------- 1. 加载model并反序列化 -------------------------------
//     TRTLogger logger;
//     auto engine_data = load_file(engine_path);
//     nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
//     nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
//     if(engine == nullptr){
//         printf("Deserialize cuda engine failed.\n");
//         runtime->destroy();
//         return;
//     }

//     nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
//     cudaStream_t stream = nullptr;
//     cudaStreamCreate(&stream);


//     // ------------------------------- 2. 输入与输出 -------------------------------
//     float input_data_host[] = {
//         // batch 0
//         1,   1,   1,
//         1,   1,   1,
//         1,   1,   1,

//         // batch 1
//         -1,   1,   1,
//         1,   0,   1,
//         1,   1,   -1
//     };
//     float* input_data_device = nullptr;

//     // 3x3输入，对应3x3输出
//     int ib = 2;
//     int iw = 3;
//     int ih = 3;
//     const int output_size = 2 * 3 * 3;  // 使用常量定义数组大小
//     float output_data_host[output_size];
//     float* output_data_device = nullptr;
//     cudaMalloc(&input_data_device, sizeof(input_data_host));
//     cudaMalloc(&output_data_device, sizeof(output_data_host));
//     cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);


//     // ------------------------------- 3. 推理 -------------------------------
//     // 明确当前推理时，使用的数据输入大小
//     execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw));
//     float* bindings[] = {input_data_device, output_data_device};
//     bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
//     cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
//     cudaStreamSynchronize(stream);


//     // ------------------------------- 4. 输出结果 -------------------------------
//     for(int b = 0; b < ib; ++b){
//         printf("batch %d. output_data_host = \n", b);
//         for(int i = 0; i < iw * ih; ++i){
//             printf("%f, ", output_data_host[b * iw * ih + i]);
//             if((i + 1) % iw == 0)
//                 printf("\n");
//         }
//     }

//     printf("Clean memory\n");
//     cudaStreamDestroy(stream);
//     cudaFree(input_data_device);
//     cudaFree(output_data_device);
//     execution_context->destroy();
//     engine->destroy();
//     runtime->destroy();
// }
