#include "onnx_inference.h"
#include <chrono>


void softmax(std::vector<float>& values) {
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    
    for (float& val : values) {
        val = std::exp(val - max_val);
        sum += val;
    }
    
    for (float& val : values) {
        val /= sum;
    }
}

Result onnx_inference(const char* model_path, const float* image_data, int height, int width)
{
    // === 1. 初始化环境 ===
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeDemo");

    // === 2. 配置会话选项 ===
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);  // 设置线程数
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // === 3. 加载模型 ===
    std::wstring wmodel_path = std::wstring(model_path, model_path + strlen(model_path));
    Ort::Session session(env, wmodel_path.c_str(), session_options);

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
    std::vector<int64_t> input_dims = {
        input_shape[0] == -1 ? 1 : input_shape[0],
        input_shape[1] == -1 ? 1 : input_shape[1],
        input_shape[2] == -1 ? height : input_shape[2],
        input_shape[3] == -1 ? width : input_shape[3]
    };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(image_data),
        input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3], 
        input_dims.data(), 
        input_dims.size()
    );

    // === 6. 执行推理 ===
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto start_time = std::chrono::high_resolution_clock::now();
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, 
        &input_tensor, 
        1, 
        output_names, 
        1
    );
    auto end_time = std::chrono::high_resolution_clock::now();

    // === 7. 处理输出 ===
    Result result;
    result.inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count() / 1000.0f;

    if (output_tensors.size() > 0 && output_tensors.front().IsTensor()) {
        float* probabilities = output_tensors.front().GetTensorMutableData<float>();
        size_t num_classes = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

        result.probabilities.assign(probabilities, probabilities + num_classes);

        // 应用 softmax
        softmax(result.probabilities);

        // 获取最高概率类别
        auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
        result.top_class_id = std::distance(result.probabilities.begin(), max_it);
        result.top_probability = *max_it;
    }

    // === 8. 清理资源 ===
    allocator.Free(input_name.release());
    allocator.Free(output_name.release());

    return result;
}