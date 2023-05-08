import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import models


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 计算特征图中每个位置的重要性权重
        weights = self.conv(x)
        weights = f.sigmoid(weights)

        # 对特征图的每个切片进行加权
        x = x * weights

        # 返回加权后的特征图
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # 空间注意力
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_out = self.sigmoid(avg_out + max_out)

        # 空间注意力
        spatial_out = self.conv1(x)
        spatial_out = self.bn1(spatial_out)
        spatial_out = self.sigmoid2(self.bn2(self.conv2(spatial_out)))
        spatial_out = self.conv3(spatial_out)
        spatial_out = self.sigmoid2(spatial_out)

        # 组合通道和空间注意力
        x = x * channel_out * spatial_out

        return x

class BDense201_fc(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_fc, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).features
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)
        # 将卷积层参数的requires_grad属性设为False，即冻结其参数
        for parameter in self.conv.parameters():
            parameter.requires_grad = False
        # 对全连接层参数进行初始化
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias, val=0)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out

class BDense201_all(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_all, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=None).features
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out

class BDense201_fc_Attention(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_fc_Attention, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).features
        # 定义注意力机制层
        self.attention = AttentionBlock(1920)
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)
        # 将卷积层参数的requires_grad属性设为False，即冻结其参数
        for parameter in self.conv.parameters():
            parameter.requires_grad = False
        # 对全连接层参数进行初始化
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias, val=0)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out

class BDense201_all_Attention(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_all_Attention, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=None).features
        # 定义注意力机制层
        self.attention = AttentionBlock(1920)
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out

class BDense201_fc_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_fc_CBAM, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).features
        # 定义注意力机制层
        self.attention = CBAM(1920)
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)
        # 将卷积层参数的requires_grad属性设为False，即冻结其参数
        for parameter in self.conv.parameters():
            parameter.requires_grad = False
        # 对全连接层参数进行初始化
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias, val=0)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out

class BDense201_all_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(BDense201_all_CBAM, self).__init__()
        # 使用DenseNet201的features部分作为卷积层
        self.conv = models.densenet201(weights=None).features
        # 定义注意力机制层
        self.attention = CBAM(1920)
        # 定义全连接层，输入维度为1920*1920，输出维度为num_classes
        self.fc = nn.Linear(1920 * 1920, num_classes)

    def forward(self, x):
        # 前向传播，先经过卷积层，再进行平均池化，最后将输出展平为二维张量
        x = self.conv(x)
        x = self.attention(x)
        x = torch.flatten(x, 2)
        # 进行交叉乘积操作，得到二维特征张量
        x = x.view(x.size(0), 1920, 7 * 7)
        x_t = torch.transpose(x, 1, 2)
        x = torch.bmm(x, x_t) / (7 * 7)
        x = x.view(x.size(0), 1920 * 1920)
        # 进行 The signed square root 操作
        features = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        # 进行 L2 归一化
        features = torch.nn.functional.normalize(features)

        out = self.fc(features)
        return out