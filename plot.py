# author xiaogang
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import  numpy as np
import  matplotlib.pyplot as plt


train_data = FashionMNIST(root='./dataset',
                          train=True,
                          transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(224)]),
                          download=True)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True,num_workers=4)


'''
为什么需要这样做？
Windows 的特殊性：

在 Linux/macOS 上，multiprocessing 默认使用 fork() 创建子进程，不会重新导入主模块。

但在 Windows 上，由于没有 fork()，Python 会重新运行主模块来创建子进程，如果没有 if __name__ == '__main__': 保护，就会导致无限循环。

freeze_support() 的作用：

当程序被打包成 .exe 时，multiprocessing 需要额外的初始化步骤，freeze_support() 就是用来处理这个问题的。
'''


if __name__ == '__main__':
    for step, (x, target) in enumerate(train_loader):
        if step > 0:
            break
    x = x.squeeze().numpy()
    y = target.numpy()
    class_labels = train_data.classes
    print(class_labels)
    #
    # plt.figure(figsize=(12, 5))
    # for i in np.arange(len(x)):
    #     plt.subplot(4, 16, i + 1)
    #     plt.imshow(x[i, :, :], cmap='gray')
    #     plt.title(class_labels[y[i]], size=10)
    #     plt.axis('off')
    #     plt.subplots_adjust(wspace=0.05)
    # plt.show()