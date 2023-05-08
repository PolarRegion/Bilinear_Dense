# *_*coding: utf-8 *_*
# author --Lee--

import torch
import torchvision
import torch.utils.data
from torchvision.datasets import ImageFolder

# 导入自定义模块 GetStat
from utils.GetStat import GetStat

# 从 opt.py 中导入命令行参数解析器
from opt import parse_opt

# 解析命令行参数
opt = parse_opt()

# 创建训练集的数据集对象
train_dataset = ImageFolder(root=str(opt.dataset) + '/train',  # 设置训练集根目录
                            transform=torchvision.transforms.ToTensor())  # 将图片转换为 PyTorch 张量

# 计算训练集的均值和标准差
mean, std = GetStat(train_dataset)


# 定义训练集数据处理函数
def train_data_process():
    # 定义训练集的图像变换操作
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),  # 调整图像大小为 256 x 256
        torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
        torchvision.transforms.RandomCrop(size=224),  # 随机裁剪成 224 x 224
        torchvision.transforms.ToTensor(),  # 转换为张量
        torchvision.transforms.Normalize(mean, std)  # 归一化
    ])
    # 创建训练集的数据集对象
    train_data = ImageFolder(root=str(opt.dataset) + '/train',  # 设置训练集根目录
                             transform=train_transforms)  # 应用上述变换操作

    # 创建训练集数据加载器
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,  # 设置批次大小
                                               shuffle=True,  # 随机打乱数据集
                                               num_workers=8,  # 设置并行加载的进程数
                                               pin_memory=True)  # 在 CPU 内存中锁定数据，加快 GPU 加载数据的速度
    return train_loader  # 返回数据加载器对象


# 定义测试集数据处理函数
def test_data_process():
    # 定义测试集的图像变换操作
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=256),  # 调整图像大小为 256 x 256
        torchvision.transforms.RandomCrop(size=224),  # 随机裁剪成 224 x 224
        torchvision.transforms.ToTensor(),  # 转换为张量
        torchvision.transforms.Normalize(mean, std)  # 归一化
    ])
    # 创建测试集的数据集对象
    test_data = ImageFolder(root=str(opt.dataset) + '/test',  # 设置测试集根目录
                            transform=test_transforms)  # 应用上述变换操作

    # 创建测试集数据加载器
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=opt.batch_size,  # 设置批次大小
                                              shuffle=False,  # 不打乱数据集
                                              num_workers=8,  # 设置并行加载的进程数
                                              pin_memory=True)  # 在 CPU 内存中锁定数据，加快 GPU 加载数据的速度
    return test_loader  # 返回数据加载器对象


# 主函数测试
if __name__ == '__main__':
    train_data_process()  # 生成训练集数据加载器
    test_data_process()  # 生成测试集数据加载器
