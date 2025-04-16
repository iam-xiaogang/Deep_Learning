# author xiaogang
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader,random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,padding=2)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=400, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.pool1(x)
        x = self.sig(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x



def train_dataset_process():
    dataset = FashionMNIST(root='./data', train=True, transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()]),download=True)
    train_data, val_data = random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])
    train_data_loader = DataLoader(train_data,batch_size=12,shuffle=True,num_workers=4)
    valid_data_loader = DataLoader(val_data,batch_size=12,shuffle=True,num_workers=4)
    return train_data_loader, valid_data_loader


def train_model_process(model,train_data_loader,valid_data_loader,epochs:int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")# 选择设备。gpu or cpu
    criterion = nn.CrossEntropyLoss()# 设置损失函数 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)# 设置优化器，学习率为0.001
    model = model.to(device)# 将model放入设备中
    best_model_dict = copy.deepcopy(model.state_dict())# 保存当前model的参数，w b

    train_loss_all = []
    train_acc_all = []
    valid_loss_all = []
    valid_acc_all = []
    best_acc = 0.0
    # 初始化变量
    since = time.time()
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        train_loss = 0.0
        train_acc = 0
        valid_loss = 0.0
        valid_acc = 0
        train_num = 0
        valid_num = 0
        # 初始化变量
        for index,(data,label) in enumerate(train_data_loader):
            data = data.to(device)
            label = label.to(device)
            model.train()# 调节为训练模式
            optimizer.zero_grad()# 将梯度设置为0
            output = model(data)# 输入数据，得到对应数据中输出值的预测
            pre_lab = torch.argmax(output, dim=1)# 找到每一行中最大值的下标
            loss = criterion(output, label)# 计算损失函数
            loss.backward()# 反向传播计算梯度
            optimizer.step()#根据梯度，选择梯度最小的值对应的模型参数
            train_loss += loss.item()*data.size(0)# 累加计算loss值，data.size(0)这里数据会是(batch_size,channel,height,width),loss.item()指的是当前batch中平均loss值
            train_acc += torch.sum(pre_lab==label.data)# 计算当前 batch 中预测正确的样本数量（pre_lab == label.data 为 True 的位置）并累加到 train_acc 中
            train_num += data.size(0)# 累加所有的batch数量，为后面的计算loss以及acc做准备
        for index,(data,label) in enumerate(valid_data_loader):
            data = data.to(device)
            label = label.to(device)
            model.eval()
            output = model(data)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, label)
            valid_loss += loss.item()*data.size(0)
            valid_acc += torch.sum(pre_lab==label.data)
            valid_num += data.size(0)
        train_loss = train_loss/train_num# # 每个 epoch 中所有样本的累计损失除以样本数，得到该 epoch 的平均 loss
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc.double().item()/train_num)## 当前 epoch 中预测正确的总样本数除以总样本数，得到该 epoch 的准确率
        valid_loss = valid_loss/valid_num
        valid_loss_all.append(valid_loss)
        valid_acc_all.append(valid_acc.double().item()/valid_num)
        if valid_acc_all[-1] > best_acc:
            best_acc = valid_acc_all[-1]
            best_model_dict = copy.deepcopy(model.state_dict())
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, valid_loss_all[-1], valid_acc_all[-1]))
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))
    model.load_state_dict(best_model_dict)
    torch.save(model.state_dict(),'./model.pth')
    pd_data = pd.DataFrame({"epoch":range(epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":valid_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":valid_acc_all,})
    return pd_data
def test_model_process(model, test_dataloader):
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output= model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)
def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


def myprint_function(function_name, split_line_length, rounds=None, *args, **kwargs):
    print(function_name+str(kwargs[rounds]))
    print(split_line_length*'-')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
    train_data_loader, valid_data_loader = train_dataset_process()
    model1 = LeNet()
    model1=model1.to(device)
    data = train_model_process(model1,train_data_loader,valid_data_loader,epochs=10)
    matplot_acc_loss(data)
    model2 = LeNet()
    model2.load_state_dict(torch.load('./model.pth'))
    test_model_process(model2,valid_data_loader)