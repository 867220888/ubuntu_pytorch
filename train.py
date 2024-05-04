import torchvision
from torch.utils.data import DataLoader
from model import *


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print('训练数据集的长度为：{}'.format(train_data_size))
print('测试数据集的长度为：{}'.format(train_data_size))

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()


# 定义优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 训练的次数
total_train_step = 0
# 测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i + 1))

    # 训练
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        # 计算损失
        loss = loss_fn(outputs, targets)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()

        print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        total_train_step += 1

with torch.no_grad():
    # 测试
    for data in test_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        print("测试次数：{}, Loss: {}".format(total_test_step, loss.item()))
        total_test_step += 1