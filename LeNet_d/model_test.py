# author xiaogang
import copy
import time
from sys import flags

import torch
# from torch._C.cpp import nn
from torch import nn
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

from LeNet_d.model import LeNet


def test_data_process():
    test_set = FashionMNIST(root='../dataset',
                              train=False,
                              download=True,
                              transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(28)]))
    # train_data,verify_data = random_split(train_set,[round(0.8*len(train_set)),round(0.2*len(train_set))])
    test_loder = DataLoader(test_set,batch_size=1,shuffle=True,num_workers=4)
    # verify_loder = DataLoader(verify_data,batch_size=64,shuffle=True,num_workers=4)
    return test_loder

def test_model(model,test_dataloader):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    test_correct = 0
    test_num = 0
    with torch.no_grad():
        for data,label in test_dataloader:
            data,label = data.to(device),label.to(device)
            model.eval()
            output = model(data)
            pre_lab = torch.argmax(output,dim=1)
            test_correct += torch.sum(pre_lab == label.data)
            test_num += data.size(0)
    test_acc = test_correct.double().item() / test_num
    print("Test Accuracy: {:.3f}".format(test_acc))

def test_value_model(model,test_dataloader):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for data,label in test_dataloader:
            data,label = data.to(device),label.to(device)
            model.eval()
            output = model(data)
            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab.item()
            really = label.item()
            print('预测值',classes[result],'真实值',classes[really])
            # print('预测值',result,'真实值',really)



if __name__ == '__main__':
    net = LeNet()

    net.load_state_dict(torch.load('./save_model/model.pth'))
    test_loder = test_data_process()
    # test_model(net,test_loder)
    test_value_model(net,test_loder)