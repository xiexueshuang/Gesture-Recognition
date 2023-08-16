import os
import torch
import torch.nn as nn
# import einops
from torchvision import transforms
import torchvision
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from myDataset import myDataset, testDataset
import glob
import torch.nn.functional as F
from torchvision.models import resnet152, ResNet152_Weights

x_dir = r"train_xd_csv"
y_dir = r'test_xd_csv'
# # 读取数据集
train_dataset = myDataset(data_dir=x_dir)
test_dataset = testDataset(data_dir=y_dir)

# # 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=0)

# 模型搭建
model = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(in_features=2048, out_features=65, bias=True)
# print(model)

# 超参数设置
learning_rate = 1e-4
epochsize = 50

# 构建模型优化器
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda:0')
print(device)

model = model.to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)

# 训练过程
def train(epoch, model, criteon, optimizer):
    model.train()
    # corrects = 0
    # train_num = 0
    total_connect = 0  # 总的正确个数
    total_num = 0  # 总的当前测试个数
    for batchidx, (data, label) in enumerate(train_dataloader):
        # print(data.shape)
        # print(type(data))
        # data = data.view(-1,28,28)

        data = data.to(device)

        label = label.to(device)
        category = model(data)
        pred = category.argmax(dim=1)
        total_connect += torch.eq(pred, label).detach().float().sum().item()
        total_num += data.size(0)
        # print(category)
        # print(category.size())
        # pre_lab = torch.argmax(category, 1)
        # print(pred)
        # 计算损失
        label = label.long()
        # print(label)
        # print(label.size())
        loss = criteon(category, label)

        # 反向更新训练
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # corrects += torch.sum(pre_lab == label.data)
        # print(data.size(0))
        # train_num += data.size(0)
        if batchidx % 10 == 0:
            # print(pred)
            print("[{}/{}] loss：{}".format(batchidx, len(train_dataloader), loss.item()))
    scheduler.step()
    # print(scheduler.get_lr())
    # 计算一次训练之后计算率
    acc = total_connect / total_num
    print('lr:', scheduler.get_last_lr(), 'epoch:', epoch + 1, 'train_acc:', acc)
    # print(epoch, 'loss:', loss.item())


# 测试过程
def eval(epoch, model):
    model.eval()
    with torch.no_grad():
        total_connect = 0  # 总的正确个数
        total_num = 0  # 总的当前测试个数

        for (data, label) in test_dataloader:
            data = data.to(device)
            label = label.to(device)
            category = model(data)

            pred = category.argmax(dim=1)
            # _, pred = category.max(dim=1)

            total_connect += torch.eq(pred, label).detach().float().sum().item()
            total_num += data.size(0)

        # 计算一次训练之后计算率
        acc = total_connect / total_num
        print('epoch:', epoch + 1, 'test_acc:', acc)

        # 保存网络结构
        torch.save(model.state_dict(), 'resnet152_x.pth')


def main():
    for epoch in range(epochsize):
        train(epoch, model, criteon, optimizer)
        print('第{}轮训练结束'.format(epoch + 1))
        if (epoch + 1) >= 16:
            torch.save(model.state_dict(), 'resn152_epoch_x.pth')
            if (epoch + 1) % 2 == 0:
                eval(epoch, model)

if __name__ == "__main__":
    main()
