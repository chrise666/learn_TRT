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
        x = x.view(-1, int(x.numel() // x.size(0)))
        x = self.fc(x)
        return x

    
class FullyConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 编码器部分（下采样）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 保持空间维度
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 下采样到1/2
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 下采样到1/4
        )
        
        # 中间转换层
        self.mid_conv = nn.Conv2d(128, 256, kernel_size=1)  # 1x1卷积调整通道数
        
        # 解码器部分（上采样）
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # 上采样到1/2
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, 
                              padding=1, output_padding=1),  # 上采样到原尺寸
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)  # 最终分类卷积
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.mid_conv(x)
        x = self.decoder(x)
        return x