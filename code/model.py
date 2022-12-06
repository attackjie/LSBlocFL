from cgitb import reset
from pyexpat import model
from traceback import print_tb
import tenseal as ts
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from time import time
from torch import dropout, nn,functional
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import argparse

import torch.nn.functional as F

vgg16 = models.vgg16(pretrained = True)
vgg = vgg16.features
for param in vgg.parameters():
    param.requires_grad_(False)

class alxnet(nn.Module):
    def __init__(self):
        super(alxnet,self).__init__()
        self.cov1=nn.Conv2d(3,24,kernel_size=3,stride=1,)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.cov2=nn.Conv2d(24,32,kernel_size=3,padding=2)
        self.cov3=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.cov4=nn.Conv2d(64,96,kernel_size=3,padding=1)
        self.cov5=nn.Conv2d(96,64,kernel_size=3,padding=1)
        self.fc1=nn.Linear(64,128)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout()
    def forward(self,x):
        x=self.relu(self.cov1(x))
        x=self.pool1(x)
        x=self.relu(self.cov2(x))
        x=self.pool1(x)
        x=self.relu(self.cov3(x))
        x=self.pool1(x)
        x=self.relu(self.cov4(x))
        x=self.pool1(x)
        x=self.relu(self.cov5(x))
        x=self.pool1(x)
        x=torch.flatten(x,start_dim=1)
        x=self.drop(x)
        x=self.relu(self.fc1(x))
        x=self.drop(x)
        x=self.fc2(x)
        return x


class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.cov1=nn.Conv2d(3,64,kernel_size=3,stride=1, padding=1)
        self.cov2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)

        self.cov3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.cov4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)

        self.cov5=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.cov6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.cov7=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2)

        self.cov8=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=1)
        self.cov9=nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
        self.cov10=nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2)

        self.cov11=nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
        self.cov12=nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
        self.cov13=nn.Conv2d(512,512,kernel_size=3,stride=1, padding=1)
        self.pool5=nn.MaxPool2d(kernel_size=2,stride=2)



        self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)
        self.relu=nn.ReLU(inplace=True)
        self.drop=nn.Dropout(p=0.5)
    def forward(self,x):
        x=self.relu(self.cov1(x))
        x=self.relu(self.cov2(x))
        x=self.pool1(x)

        x=self.relu(self.cov3(x))
        x=self.relu(self.cov4(x))
        x=self.pool2(x)

        x=self.relu(self.cov5(x))
        x=self.relu(self.cov6(x))
        x=self.relu(self.cov7(x))
        x=self.pool3(x)


        x=self.relu(self.cov8(x))
        x=self.relu(self.cov9(x))
        x=self.relu(self.cov10(x))
        x=self.pool4(x)

        x=self.relu(self.cov11(x))
        x=self.relu(self.cov12(x))
        x=self.relu(self.cov13(x))
        x=self.pool5(x)
        x = x.view(-1, 512)
        x=self.drop(x)
        x=self.relu(self.fc1(x))
        x=self.drop(x)
        x=self.relu(self.fc2(x))
        x=self.drop(x)
        x=self.relu(self.fc3(x))
        return x
class VGGNet16(nn.Module):
    def __init__(self):
        super(VGGNet16, self).__init__()

        self.Conv1 = nn.Sequential(
            # CIFAR10 数据集是彩色图 - RGB三通道, 所以输入通道为 3, 图片大小为 32*32
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            # inplace-选择是否对上层传下来的tensor进行覆盖运算, 可以有效地节省内存/显存
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        """self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )"""

        """self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )"""

        # 全连接层
        self.fc = nn.Sequential(

            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            # 使一半的神经元不起作用，防止参数量过大导致过拟合
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 四个卷积层
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        #x = self.Conv4(x)
        #x = self.Conv5(x)

        # 数据平坦化处理，为接下来的全连接层做准备
        x=torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x
class BasicBlock(nn.Module):
    expansion = 1
    #
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # 1x1卷积
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # 3x3卷积
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # 1x1卷积
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #堆叠
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
        #7X7卷积核，stride=2,padding=3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#3x3卷积核，stride(步距)=2

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        #条件 步长不为1（第一层是没有downsample的） 或者 输入channel不等于 channel*4 倍 怎么理解
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # range 从第一个开始，因为第0个是虚线的残差层，并且在上面已经实现
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

resnet = resnet34(num_classes=5)

 
if __name__ == '__main__':
    
    size = 32
    y = torch.randn(16, 3, size, size)
  
    model=VGGNet16()
    x=model(y)
    print(x)
    