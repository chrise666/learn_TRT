#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

/**
 * @brief 推理结果
 */
struct Result
{
    std::vector<float> probabilities; // 各类别概率
    int top_class_id;                // 最高概率类别ID
    float top_probability;           // 最高概率值
    float inference_time;            // 推理时间 (ms)
};

/**
 * @brief 执行单图像推理
 * @param model_path 模型路径
 * @param image_data 图像数据
 * @param height 图像高度
 * @param width 图像宽度
 * @return 推理结果
 */
Result onnx_inference(const char* model_path, const float* image_data, int height, int width);

#endif