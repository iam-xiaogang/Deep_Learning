# author xiaogang
import torch
from torch import nn
from torchsummary import  summary


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


# lenet = LeNet()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))
