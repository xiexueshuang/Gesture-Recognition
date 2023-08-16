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
from torchvision.models import resnet152, ResNet152_Weights, VGG16_Weights, vgg16, regnet_y_1_6gf, \
    RegNet_Y_1_6GF_Weights, RegNet_Y_32GF_Weights, regnet_y_32gf

x_dir = r"train_xd_csv"
y_dir = r'test_xd_csv'
# # 读取数据集
train_dataset = myDataset(data_dir=x_dir)
# train_dataset = testDataset(data_dir=x_dir)
test_dataset = testDataset(data_dir=y_dir)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=0)

# 超参数设置
learning_rate = 1e-4
epochsize = 50

# 构建模型优化器
# print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda:6')
print(device)

model = torchvision.models.regnet_y_32gf(weights=RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1)
# model = torchvision.models.regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
model.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.fc = nn.Linear(in_features=888, out_features=65, bias=True)  # regnet_y_1_6gf
model.fc = nn.Linear(in_features=3712, out_features=65, bias=True)  # regnet_y_32gf
# print(model)
# model = torchvision.models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
# # model = torchvision.models.vision_transformer.VisionTransformer(image_size=() num_classes = 65)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# model.fc = nn.Linear(in_features=2048, out_features=65, bias=True)

# model = model.to(torch.float32)
model = model.to(device)
criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
max_acc = 0


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
        # print(data.size())   # torch.Size([64, 400, 169])

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
        if batchidx % 20 == 0:
            # print(pred)
            print("[{}/{}] loss：{}".format(batchidx, len(train_dataloader), loss.item()))

    scheduler.step()
    # print(scheduler.get_lr())
    # 计算一次训练之后计算率
    acc = total_connect / total_num
    print('lr:', scheduler.get_last_lr(), 'epoch:', epoch + 1, 'train_acc:', acc)

    # 测试过程


def eval(epoch, model):
    global max_acc
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
        if max_acc < acc:
            max_acc = acc
            print('max_acc:', max_acc)
            torch.save(model.state_dict(), 'regnet_32_noise.pth')


def main():
    for epoch in range(epochsize):
        train(epoch, model, criteon, optimizer)
        print('第{}轮训练结束'.format(epoch + 1))
        if (epoch + 1) >= 18 and (epoch + 1) % 2 == 0:
            eval(epoch, model)


if __name__ == "__main__":
    main()

