#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

bool onnx_inference(const char* model_path) {
    // === 1. 初始化环境 ===
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeDemo");

    // === 2. 配置会话选项 ===
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);  // 设置线程数
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // === 3. 加载模型 ===
    std::wstring widestr = std::wstring(model_path, model_path + strlen(model_path));
    Ort::Session session(env, widestr.c_str(), session_options);

    // === 4. 准备输入输出 ===
    // 获取输入/输出信息
    Ort::AllocatorWithDefaultOptions allocator;
    
    // 输入信息
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = input_tensor_info.GetShape();
    auto input_name = session.GetInputNameAllocated(0, allocator);

    // 输出信息
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_tensor_info.GetShape();

    // === 5. 创建输入 Tensor ===
    // 示例：创建一个 1x3x224x224 的 float 输入
    std::vector<float> input_data(1 * 1 * 128 * 128, 0.5f);  // 填充测试数据
    std::vector<int64_t> input_dims = {1, 1, 128, 128};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_data.data(), 
        input_data.size(), 
        input_dims.data(), 
        input_dims.size()
    );

    // === 6. 执行推理 ===
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );

    // === 7. 处理输出 ===
    if (output_tensors.size() > 0 && output_tensors.front().IsTensor()) {
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::cout << "Output shape: ";
        for (auto dim : output_shape) std::cout << dim << " ";
        std::cout << "\nFirst 10 values: ";
        for (int i = 0; i < 10 && i < output_size; ++i) {
            std::cout << output_data[i] << " ";
        }
    }

    // === 8. 清理资源 ===
    allocator.Free(input_name.release());
    allocator.Free(output_name.release());

    return true;
}