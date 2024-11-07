# 导入必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import *
import numpy as np
import sys
# 定义dense block中的dense layer
class _DenseLayer(nn.Module):
    # 构造函数，接收输入通道数num_input_features，输出通道数growth_rate，卷积层的缩放比例bn_size
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        # 定义第一个卷积层，包括BN层、ReLU激活函数和1x1卷积层
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        # 定义第二个卷积层，包括BN层、ReLU激活函数和3x3卷积层
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    # 定义前向传播函数
    def forward(self, x):
        # BN+ReLU+1x1卷积
        out = self.conv1(x)
        # BN+ReLU+3x3卷积
        out = self.conv2(out)
        # 将输入和输出进行拼接后返回结果
        return torch.cat([x, out], 1)
# 定义dense block
class _DenseBlock(nn.Module):
    # 构造函数，包含密集连接层的数量num_layers，输入通道数num_input_features，输出通道数growth_rate，卷积层的缩放比例bn_size
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size):
        super().__init__()
        
        # 保存密集连接层的列表
        layers = []
        # 构建num_layers个密集连接层
        for i in range(num_layers):
            # 构建一个密集连接层，其中输入通道数为num_input_features + i * growth_rate逐层递增
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            # 将构建好的密集连接层添加到列表中保存
            layers.append(layer)
        # 将所有密集连接层封装到Sequential中保存为block
        self.block = nn.Sequential(*layers)
        
    # 定义前向传播函数
    def forward(self, x):
        # 经过当前block输出即可
        return self.block(x)
# 定义dense block之间的transition layer
class _Transition(nn.Module):
    # 构造函数，输入通道数num_input_features，输出通道数num_output_features
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        # 定义一个转换层，用于降维和调整特征图的size，包含BN+ReLU+1x1卷积+平均池化层
        self.trans = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    # 定义前向传播函数
    def forward(self, x):
        # 经过转换层后输出即可
        return self.trans(x)
# 定义DenseNet的网络结构
class DenseNet(nn.Module):
    # 构造函数，包含dense block的数量block_config，输入通道数num_input_features，输出通道数growth_rate，
    #           卷积层的缩放比例bn_size和类别数num_classes
    def __init__(self, block_config, num_init_features=64, growth_rate=32, bn_size=4, num_classes=5):

        super().__init__()

        # 第一部分，7x7卷积+BN+ReLU+最大池化层
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 下面依次定义dense block和transition layer，对应第二部分和第三部分
        num_features = num_init_features # 记录通道数
        layers = [] # 网络结构保存列表
        # 遍历每层dense block的数量列表
        for i, num_layers in enumerate(block_config):
            # 创建dense block，其中包含num_layers个dense layer
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, 
                                growth_rate=growth_rate, bn_size=bn_size)
            layers.append(block)
            num_features = num_features + num_layers * growth_rate # 更新特征图维度
            # 如果不是最后一个dense block，则添加一个transition layer，特征图维度除以2
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                layers.append(trans)
                num_features = num_features // 2
        # 添加一个BN层
        layers.append(nn.BatchNorm2d(num_features))
        # 调用nn.Sequential完成第二部分和第三部分
        self.denseblock = nn.Sequential(*layers)

        # 第四部分，全连接层
        self.classifier = nn.Linear(num_features, num_classes)

    # 定义前向传播函数
    def forward(self, x):
        # 第一部分
        features = self.features(x)
        # 第二、三部分
        features = self.denseblock(features)
        # ReLU
        out = F.relu(features, inplace=True)
        # 第四部分，平均池化+全连接层
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        # 输出
        return out
def densenet201(num_classes=1000):
    return DenseNet(block_config=(6, 12, 48, 32), num_init_features=64, 
                    growth_rate=32, num_classes=num_classes)