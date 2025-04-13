# author xiaogang
# author xiaogang


import copy
import time
import os
import torch
# from torch._C.cpp import nn
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

# from LeNet_d.model import LeNet
# from AlexNet.model import AlexNet
from Vgg.model import VGG16


def tarin_verify_data_process():
    train_set = FashionMNIST(root='../dataset',
                              train=True,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(227)]))
    train_data,verify_data = random_split(train_set,[round(0.8*len(train_set)),round(0.2*len(train_set))])
    train_loder = DataLoader(train_data,batch_size=32,shuffle=True,num_workers=4)
    verify_loder = DataLoader(verify_data,batch_size=32,shuffle=True,num_workers=4)
    return train_loder,verify_loder


def train_model_process(model,train_loder,verify_loder,num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        train_loss = 0.0 # 训练集损失函数
        train_corrects = 0 # 训练集损失函数
        val_loss = 0.0 # 验证集损失函数
        val_corrects = 0.0 # 验证集损失函数
        train_num = 0 #训练集样本数量
        val_num = 0 # 验证集样本数量
        # 开始对每一个batch计算
        for batch_idx, (data, target) in enumerate(train_loder):
            data, target = data.to(device), target.to(device)# 特征放入训练设备
            model.train()# 设置为训练模式
            output = model(data)# 对每一个batch中的输出对应的预测
            pre_leb = torch.argmax(output, dim=1)# 查找每一行中最大值对应的下标
            loss = criterion(output, target)# 计算损失函数
            optimizer.zero_grad()# 将梯度初始化为0
            loss.backward()# 反响传播计算
            optimizer.step()# 根据反向传播的梯度信息来更新网络的参数
            train_loss += loss.item()*target.size(0)# 对损失函数进行累加
            train_corrects += torch.sum(pre_leb==target.data)# 如果预测正确，则train_corrects+1
            train_num += target.size(0)# 训练的样本数量
        for batch_idx, (data, target) in enumerate(train_loder):
            data, target = data.to(device), target.to(device)# 特征放入训练设备
            model.eval()# 设置为训练模式
            output = model(data)# 对每一个batch中的输出对应的预测
            pre_leb = torch.argmax(output, dim=1)# 查找每一行中最大值对应的下标
            loss = criterion(output, target)# 计算损失函数
            # optimizer.zero_grad()# 将梯度初始化为0
            # loss.backward()# 反响传播计算
            # optimizer.step()# 根据反向传播的梯度信息来更新网络的参数
            val_loss += loss.item()*target.size(0)# 对损失函数进行累加
            val_corrects += torch.sum(pre_leb==target.data)# 如果预测正确，则train_corrects+1
            val_num += target.size(0)# 训练的样本数量
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch+1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))
        if val_loss_all[-1] > best_acc:
            best_acc = val_loss_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_use = time.time()-since
        print('训练时间是{:.0f}m'.format(time_use/60))
    # model.load_state_dict(best_model_wts)
    os.makedirs('./save_model',exist_ok=True)
    torch.save(best_model_wts,'./save_model/model.pth')
    train_process =pd.DataFrame(data={
            'epoch':range(1,num_epochs+1),
            'train_loss':train_loss_all,
            'train_acc':train_acc_all,
            'val_loss':val_loss_all,
            'val_acc':val_acc_all,

        })
    return train_process

def matplotlib_imshow(data):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(data['epoch'],data.train_loss,'r-',label='train loss')
    plt.plot(data['epoch'],data.val_loss,'b-',label='val loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data.train_acc, 'r-', label='train acc')
    plt.plot(data['epoch'], data.val_acc, 'b-', label='val acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    net = VGG16()

    train_loder,verify_loder = tarin_verify_data_process()
    train_process_data = train_model_process(net,train_loder,verify_loder,num_epochs=20)
    matplotlib_imshow(train_process_data)





