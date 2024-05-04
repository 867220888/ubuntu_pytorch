import torchvision
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集的长度为：{}'.format(train_data_size))
print('测试数据集的长度为：{}'.format(train_data_size))
#
# # 利用 DataLoader 加载数据集
# train_dataloader = DataLoader(train_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)
#
# # 搭建神经网络
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*4*4, 64),
#             nn.Linear(64, 10)
#         )