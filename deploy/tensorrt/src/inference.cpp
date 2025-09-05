#include "trt_unit.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;


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

bool inference(TRTLogger& logger, const char* engine_path, float input_data_host[], float output_data_host[]) {
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

    // TODO: 这里需要改为自动获取输入张量的形状
    int ib = 2;
    int iw = 3;
    int ih = 3;

    for (int32_t i = 0; i < engine->getNbIOTensors(); i++)
    {
      const char* name = engine->getIOTensorName(i);
      TensorIOMode mode = engine->getTensorIOMode(name);

      // 设置输入输出张量地址      
      if (mode == TensorIOMode::kINPUT) {
          std::cout << "输入张量: " << name << std::endl;
          // TODO: 这里需要改为自动设置输入张量的形状
          execution_context->setInputShape(name, Dims4(ib, 1, ih, iw));
          execution_context->setInputTensorAddress(name, input_data_device);
      } else if (mode == TensorIOMode::kOUTPUT) {
          std::cout << "输出张量: " << name << std::endl;
          execution_context->setOutputTensorAddress(name, output_data_device);
      }
    }

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    // 确保所有动态输入尺寸都已指定
    if (!execution_context->allInputDimensionsSpecified())
    {
        return false;
    }
    bool success = execution_context->enqueueV3(stream);

    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ------------------------------- 4. 输出结果 -------------------------------
    // TODO: 通用的输出结果方式
    for(int b = 0; b < ib; ++b){
        printf("batch %d. output_data_host = \n", b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f, ", output_data_host[b * iw * ih + i]);
            if((i + 1) % iw == 0)
                printf("\n");
        }
    }
    // printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 5. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);

    return success;
}
