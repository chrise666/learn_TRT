import torch.nn as nn


# 定义一个简单卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # ReLU激活
        self.relu = nn.ReLU()
        # 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc = nn.Linear(16 * 16 * 16, 10)  # 假设输入图像为32x32

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)  # 展平，适合ONNX和TensorRT
        x = self.fc(x)
        return x