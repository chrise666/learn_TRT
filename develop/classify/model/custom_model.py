import torch
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

# 定义一个简单全卷积神经网络    
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

# 用于形状分类的卷积神经网络
class ShapeClsNet(nn.Module):
    def __init__(self, num_class):
        super(ShapeClsNet, self).__init__()
        
        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 自适应池化层（处理任意尺寸输入）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 特征提取
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 自适应池化（统一特征图尺寸）
        x = self.adaptive_pool(x)
        
        # 分类
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.softmax(x)

        return x

class BasicBlock(nn.Module):
    """基础残差块（适用于浅层网络）"""
    expansion = 1  # 通道扩展系数

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径连接（当维度不匹配时使用1x1卷积调整）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = nn.ReLU()(out)
        return out

class BottleneckBlock(nn.Module):
    """瓶颈残差块（适用于深层网络）"""
    expansion = 4  # 通道扩展系数

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = out_channels  # 中间层通道数
        
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        # 3x3卷积（主路径）
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        # 1x1卷积升维
        self.conv3 = nn.Conv2d(
            mid_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = nn.ReLU()(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block_type, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建残差层
        self.layer1 = self._make_layer(block_type, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_type, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block_type, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_type, 512, layers[3], stride=2)
        
        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, block_type, out_channels, blocks, stride):
        layers = []
        # 第一个块可能有下采样
        layers.append(block_type(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block_type.expansion
        
        # 后续块
        for _ in range(1, blocks):
            layers.append(block_type(self.in_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes)