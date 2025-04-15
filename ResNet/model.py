# author xiaogang
import torch
from torch import nn
from torchsummary import summary


class Resiual(nn.Module):
    def __init__(self, in_channels, num_channels,use_1conv=False,stride=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y = self.relu(x + y)
        return y

class ResNet(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=7,padding=3,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        )
        self.b2 = nn.Sequential(
            r(64, 64, use_1conv=False,stride=1),
            r(64, 64, use_1conv=False, stride=1),
        )
        self.b3 = nn.Sequential(
            r(64, 128, use_1conv=True,stride=2),
            r(128, 128, use_1conv=False, stride=1),
        )
        self.b4 = nn.Sequential(
            r(128, 256, use_1conv=True,stride=2),
            r(256, 256, use_1conv=False, stride=1),
        )
        self.b5 = nn.Sequential(
            r(256, 512, use_1conv=True,stride=2),
            r(512, 512, use_1conv=False, stride=1),
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10),
        )
    def forward(self, x):
        y = self.b1(x)
        y = self.b2(y)
        y = self.b3(y)
        y = self.b4(y)
        y = self.b5(y)
        y = self.b6(y)
        return y

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(Resiual).to(device)
    print(summary(model,(1,224,224)))

