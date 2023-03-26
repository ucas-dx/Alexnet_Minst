# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/24 22:55
# @Function: test
from AlexNetModel import AlexModel
import torch
import torchvision
from torch.utils.data import DataLoader, SequentialSampler
import tqdm
#定义使用设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 缩放图像大小为 224*224

#创建损失函数
LossF=torch.nn.CrossEntropyLoss()#使用交叉损失熵
# 加载测试数据集
test_data = torchvision.datasets.FashionMNIST(
    root="data",              # 数据集保存路径
    train=False,               # 加载训练集
    download=True,            # 如果本地没有数据集，则从互联网下载
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize(224),#拉伸为alexnet输入图片的大小
    torchvision.transforms.ToTensor(),# 将图像转换为张量
                                              ])
)

# 给测试集创建一个数据集加载器,因为用于测试，所以batch_size=1，并且选择按顺序测试
test_dataloader = DataLoader(test_data, batch_size=1,shuffle=False,sampler=SequentialSampler(test_data))

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = AlexModel().to(device)
# 加载 train.py 里训练好的模型
model.load_state_dict(torch.load(r"model11.pth"))

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # 获取数据集的大小
    num_batches = len(dataloader)  # 获取数据集batch的个数
    test_loss, correct = 0, 0
    with torch.no_grad():  # 不需要计算梯度，验证阶段无需动训练好的模型数据
        model.eval()#和Bn联合使用
        for X, y in tqdm.tqdm(dataloader):  # 遍历数据集
            X, y = X.to(device), y.to(device)  # 转换为张量
            pred = model(X)  # 模型预测
            test_loss += loss_fn(pred, y).item()  # 计算损失，item导出单个值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # 计算正确率,取预测概率最大的那个数比较，并把一个batch内的所有预测求和
            #
    test_loss /= num_batches  # 计算平均损失，使用batch的数量，因为test_loss是一个batch为单位计算的
    correct /= size  # 计算正确率，所有的数据参与
    print(f"测试数据: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # 打印测试结果

if __name__=="__main__":
    test(test_dataloader,model,LossF)
