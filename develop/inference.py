import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

# 1. 加载ONNX模型
def load_onnx_model(model_path):
    """加载ONNX模型并创建推理会话"""
    # 设置ONNX Runtime选项
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # 创建推理会话
    providers = ['CPUExecutionProvider']  # 默认使用CPU

    # 检查GPU可用性
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("检测到GPU，将使用CUDA加速")
    
    session = ort.InferenceSession(model_path, options, providers=providers)
    
    # 打印模型输入输出信息
    input_name = session.get_inputs()[0].name
    # input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    # output_shape = session.get_outputs()[0].shape
    
    print(f"模型加载成功: {os.path.basename(model_path)}")
    # print(f"输入名称: {input_name}, 形状: {input_shape}")
    # print(f"输出名称: {output_name}, 形状: {output_shape}")
    
    return session, input_name, output_name

# 2. 图像预处理
def preprocess_image(image_path, target_size=None):
    """
    预处理单通道图像:
    1. 读取图像
    2. 转换为灰度（如果是彩色）
    3. 调整尺寸（如果需要）
    4. 归一化
    5. 转换为模型输入格式
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 直接读取为灰度图
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 保存原始图像用于显示
    original_img = img.copy()
    
    # 调整尺寸（如果指定了目标尺寸）
    if target_size is not None:
        h, w = target_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 归一化 [-1, 1] 范围
    img = img.astype(np.float32) / 255.0  # [0, 1]
    img = (img - 0.5) / 0.5  # [-1, 1]
    
    # 添加批次和通道维度 (1, 1, H, W)
    img = np.expand_dims(img, axis=0)  # 添加通道维度
    img = np.expand_dims(img, axis=0)  # 添加批次维度
    
    return original_img, img

# 3. 执行推理
def run_inference(session, input_name, output_name, input_data):
    """使用ONNX Runtime执行推理"""
    start_time = time.time()
    
    # 运行推理
    outputs = session.run([output_name], {input_name: input_data})
    
    # 获取输出结果
    predictions = outputs[0]
    inference_time = (time.time() - start_time) * 1000  # 毫秒
    
    return predictions, inference_time

# 4. 处理结果
def process_predictions(predictions, class_names):
    """处理预测结果并返回类别和置信度"""
    # 应用softmax获取概率分布
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    
    # 获取top-k结果
    top_k = 3
    top_indices = np.argsort(probabilities[0])[::-1][:top_k]
    
    # 创建结果字典
    results = []
    for idx in top_indices:
        results.append({
            "class_id": idx,
            "class_name": class_names[idx] if idx < len(class_names) else str(idx),
            "confidence": float(probabilities[0][idx])
        })
    
    # 获取最高置信度的结果
    top_result = results[0]
    
    return top_result, results

# 5. 显示结果
def display_results(original_img, top_result, all_results, target_size=None, inference_time=0):
    """可视化原始图像和预测结果"""
    # plt.figure(figsize=(12, 6))
    
    # # 显示原始图像
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_img, cmap='gray')
    # plt.title("原始图像")
    
    # # 如果调整了尺寸，显示调整后的图像
    # if target_size is not None:
    #     resized_img = cv2.resize(original_img, target_size[::-1])  # (w, h) -> (h, w)
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(resized_img, cmap='gray')
    #     plt.title(f"调整后尺寸: {target_size[1]}x{target_size[0]}")
    # else:
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(original_img, cmap='gray')
    #     plt.title("原始尺寸")
    
    # # 添加预测结果文本
    # plt.figtext(0.5, 0.01, 
    #             f"预测结果: {top_result['class_name']} ({top_result['confidence']*100:.2f}%)", 
    #             ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    # plt.tight_layout()
    # plt.show()
    
    # 打印详细结果
    print("\n预测结果:")
    print(f"  最可能类别: {top_result['class_name']} (置信度: {top_result['confidence']*100:.2f}%)")
    print("\nTop 3预测:")
    for i, res in enumerate(all_results):
        print(f"  {i+1}. {res['class_name']}: {res['confidence']*100:.2f}%")
    
    print(f"  模型推理时间: {inference_time:.2f} ms")

# 主函数
def main():
    # 配置参数
    MODEL_PATH = "E:/workspace/save/shape_classification/best.onnx"  # 替换为你的ONNX模型路径
    IMAGE_PATH = "E:/workspace/save/shape_classification/img/block.png"              # 替换为你的测试图像路径
    CLASS_NAMES = ["点状", "团状", "线状"]   # 替换为你的类别名称
    
    # 1. 加载ONNX模型
    session, input_name, output_name = load_onnx_model(MODEL_PATH)
    
    # 从模型获取输入尺寸（如果是固定尺寸）
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and input_shape[2] != "?" and input_shape[3] != "?":
        target_size = (input_shape[2], input_shape[3])  # (H, W)
        print(f"检测到固定输入尺寸: {target_size[1]}x{target_size[0]}")
    else:
        target_size = (128, 128)  # 默认尺寸或根据你的模型设置
        print(f"使用默认尺寸: {target_size[1]}x{target_size[0]}")
    
    # 2. 预处理图像
    original_img, processed_img = preprocess_image(IMAGE_PATH, target_size)
    print(f"输入数据形状: {processed_img.shape}, 数据类型: {processed_img.dtype}")
    
    # 3. 执行推理
    predictions, inference_time = run_inference(session, input_name, output_name, processed_img)
    print(f"原始输出: {predictions}")
    
    # 4. 处理预测结果
    top_result, all_results = process_predictions(predictions, CLASS_NAMES)
    
    # 5. 显示结果
    display_results(original_img, top_result, all_results, target_size, inference_time)

if __name__ == "__main__":
    main()