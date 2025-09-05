import os, random
import numpy as np
import copy
import torch
import torch.export
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization import move_exported_model_to_train, move_exported_model_to_eval

from classify.config import Config
from classify.data import *
from classify.model.custom_model import resnet50


# 模型转换
def prepare_qat_model(model_fp, checkpoint=None, device="cpu"):
    model_to_quantize = copy.deepcopy(model_fp)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        model_to_quantize.load_state_dict(state_dict, strict=False)

    # 定义量化配置
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config())

    # 准备模型进行PT2E量化
    example_inputs = (torch.randn(1, 3, 128, 128), )
    model_to_quantize = torch.export.export(model_to_quantize, example_inputs).module()
    model_prepared = prepare_pt2e(model_to_quantize, quantizer)

    return model_prepared.to(device)

# 训练函数
def train_qat_model(model, train_loader, test_loader, epochs=10, lr=0.001, device="cpu"):
    best_model_wts = copy.deepcopy(model.state_dict())

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_accuracy = 0.0
    # 训练循环
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        move_exported_model_to_train(model)  # 确保处于训练模式
        progress_bar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{epochs}]')
        for i, (inputs, labels, _) in enumerate(progress_bar):
            optimizer.zero_grad()

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': running_loss/(i+1), 
                'Acc': 100.*correct/total
            })

        # 更新学习率
        scheduler.step()

        # 评估模型
        move_exported_model_to_eval(model)
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'test accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = max(best_accuracy, accuracy)
            best_model_wts = copy.deepcopy(model.state_dict())
        print(f'best accuracy: {best_accuracy:.2f}%')

    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    # 转换为量化模型
    model_quantized = convert_pt2e(model)
    return model_quantized

# 主函数
def main():
    # 超参数
    cfg = Config()
    checkpoint = f"{cfg.save}/best.pth"
    
    os.makedirs(cfg.save, exist_ok=True)

    seed=cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 准备数据
    print("============== Preparing Data... ====================")
    datasets, classes, classes_num = build_split(cfg)

    # 创建模型
    print("============== Creating Model... ====================")
    model = resnet50(num_classes=len(classes))
    model_to_quantize = prepare_qat_model(model, checkpoint, device)

    # 模型训练
    print("============== Preparing Training... ====================")
    for f, (train_set, valid_set) in enumerate(datasets):
        train_loader = DataloaderBase(dataset=train_set, opt=cfg)
        test_loader = DataloaderBase(dataset=valid_set, opt=cfg)
        
        # 创建并训练QAT模型
        quantized_model = train_qat_model(
            model_to_quantize, train_loader, test_loader, 
            epochs=cfg.epoch, lr=cfg.lr, device=device
        )
        
    # 保存量化模型
    torch.save(quantized_model.state_dict(), f"{cfg.save}/resnet50_int8.pth")
    print("量化模型已保存")


if __name__ == '__main__':
    main()