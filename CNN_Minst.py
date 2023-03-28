# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/23 19:43
# @Function:Model_Minst

import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm,trange


import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3,stride=2, padding=1),
                                  nn.BatchNorm2d(8),
                                  nn.Conv2d(8,16,kernel_size=3,stride=2, padding=0),
                                  nn.BatchNorm2d(16),
                                  nn.Conv2d(16,32,kernel_size=3,stride=2, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(64),#4
                                  nn.ReLU()
                                  )
        self.flaten=nn.Flatten()
        self.L = nn.Sequential(nn.Linear(576,100),
                               nn.ReLU(),
                               nn.Linear(100, 10),
                                  )


    def forward(self, x):
        x=self.conv(x)
        x=self.flaten(x)
        x=self.L(x)
        return x




#超参数设置
learning_rate=0.001#学习率
epoch=40
bach_size=32
#定义使用设备
device=torch.device( "cuda" if torch.cuda.is_available() else "cpu")

#模型保存到定义的设备
model=Net().to(device)

#通过torchvision加载在线数据集
training_data = torchvision.datasets.MNIST(
    root="data",              # 数据集保存路径
    train=True,               # 加载训练集
    download=True,            # 如果本地没有数据集，则从互联网下载
    #对图片进行一系列变换
    transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()# 将图像转换为张量
                                              ])
)

test_data = torchvision.datasets.MNIST(
    root="data",              # 数据集保存路径
    train=False,               # 加载训练集
    download=True,            # 如果本地没有数据集，则从互联网下载
    transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()# 将图像转换为张量
                                              ])
)

# 创建训练和测试数据的 DataLoader
train_dataloader = DataLoader(training_data, batch_size=bach_size,shuffle=True)#shuffle=True打乱取样
test_dataloader = DataLoader(test_data, batch_size=bach_size,shuffle=False)

#创建损失函数
LossF=torch.nn.CrossEntropyLoss()#使用交叉损失熵

#使用SGD优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.5)
# 学习率每隔 10 个 epoch 变为原来的 0.5
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

def train_loop(dataloader,model,loss_fn,optimizer,epoch):#定义训练循环
    sum_loss=0
    n=0
    loss=0
    size = len(dataloader.dataset)  # 获取数据集的大小
    for batch, (X, y) in tqdm(enumerate(dataloader),total=len(dataloader),leave=True,position=0):  # 遍历数据集
        X, y = X.to(device), y.to(device)#转换为张量
        pred = model(X)  # 模型预测
        loss = loss_fn(pred, y)  # 计算损失
        # 后向传播
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        sum_loss+=loss.item()#迭代计算总的损失值
        n+=1
        """
        if batch % 99==0:
            print('Epoch{} [{}/{}({:.0f}%)]\t Loss:{:.6f}'.format(epoch+1, batch * len(X), 
                                                                  len(dataloader.dataset),
                                                                  100. * batch / len(dataloader),
                                                                  (sum_loss/n)))
        """
    print("Epoch{}的loss为：{}".format(epoch+1,sum_loss/n))


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 获取数据集的大小
    num_batches = len(dataloader)  # 获取数据集batch的个数
    test_loss, correct = 0, 0
    with torch.no_grad():  # 不需要计算梯度，验证阶段无需动训练好的模型数据
        for X, y in tqdm(dataloader):  # 遍历数据集
            X, y = X.to(device), y.to(device)  # 转换为张量
            model.eval()
            pred = model(X)  # 模型预测
            test_loss += loss_fn(pred, y).item()  # 计算损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 计算正确率,取预测概率最大的那个数比较，并把一个batch内的所有预测求和
            #
    test_loss /= num_batches  # 计算平均损失，使用batch的数量，因为test_loss是一个batch为单位计算的
    correct /= size  # 计算正确率，所有的数据参与
    print(f"测试数据: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # 打印测试结果


if __name__=="__main__":
    for t in range(epoch):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, LossF, optimizer,t)
        test_loop(test_dataloader, model, LossF)
        #lr_scheduler.step()
        torch.save(model.state_dict(), r"model.pth")  # 模型保存,如果使用BN需要配合model.eval()
    print("Model Train Done!")


