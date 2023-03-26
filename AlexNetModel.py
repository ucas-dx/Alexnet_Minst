# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/23 19:43
# @Function:Model
import torch
import torch.nn as nn
class AlexModel(nn.Module):#定义AlexNet的模型
    def __init__(self):#width_mult为输入图像的尺寸缩放比默认为1。
        super(AlexModel,self).__init__()#继承父类的初始化方法
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=2),#(224 - 11 + 2*2) / 4 + 1 = 55包含96个大小为11×11的滤波器（其实是11×11×3），卷积步长为4，因此第一层输出大小为55×55×96，padding=2；
            nn.BatchNorm2d(num_features=96,eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),#核大小为3×3、步长为2的最大池化层进行数据降采样，进而输出大小为27×27×96
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),#27 - 5 + 2*2) / 1 + 1 = 27包含256个大小为5×5滤波器，卷积步长为1，同时利用padding保证输出尺寸不变，因此该层输出大小为27×27×256
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)#通过核大小为3×3、步长为2的最大池化层进行数据降采样，进而输出大小为13×13×256
                                  )
        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=384, eps=1e-05, momentum=0.1, affine=True),
            #(13 - 3 + 1*2) / 1 + 1 = 13
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1,padding=1),# (13 - 3 + 1*2) / 1 + 1 = 13
            nn.BatchNorm2d(num_features=384, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),# (13 - 3 + 1*2) / 1 + 1 = 13
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)#out 256*6*6
        )
        #定义全连接层,用于1-10分类得分
        self.fc=nn.Sequential(#nn.Dropout(0.5),
                              nn.Linear(9216,4096),
                              nn.ReLU(inplace=True),
                              nn.Dropout(0.5),
                              nn.Linear(4096, 4096),
                              nn.ReLU(inplace=True),
                              nn.Dropout(0.5),
                              nn.Linear(4096, 1000),
                              nn.ReLU(inplace=True),
                              nn.Linear(1000, 10))

    def forward(self, x):#前向传播

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x,start_dim=1)#torch.flatten是功能函数不是类，展平为一元
        x = self.fc(x)
        return x




