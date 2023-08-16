import os
import json
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
import matplotlib.pyplot as plt


# from prettytable import PrettyTable


class ConfusionMatrix(object):
    # """
    # 注意，如果显示的图像不全，是matplotlib版本问题
    # 本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    # 需要额外安装prettytable库
    # """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            t = int(t)
            self.matrix[p, t] += 1

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.figure(figsize=(30, 30))
        plt.savefig('./test.jpg')
        plt.show()


if __name__ == "__main__":

    y_dir = r'test_xd_csv'
    # # 读取数据集
    test_dataset = testDataset(data_dir=y_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=40, num_workers=0)#, drop_last=True

    # print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)

    model = torchvision.models.regnet_y_32gf()
    # model = torchvision.models.regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
    model.stem[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # model.fc = nn.Linear(in_features=888, out_features=65, bias=True)  # regnet_y_1_6gf
    model.fc = nn.Linear(in_features=3712, out_features=65, bias=True)  # regnet_y_32gf
    model_weight_path = "./resn152_epoch_x.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model = model.to(device)

    # read class_indict
    # json_label_path = './class_indices.json'
    # assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    # json_file = open(json_label_path, 'r')
    # class_indict = json.load(json_file)

    labels = [i for i in range(65)]
    print(labels)
    confusion = ConfusionMatrix(num_classes=65, labels=labels)
    model.eval()
    with torch.no_grad():
        # for val_data in tqdm(validate_loader):
        for step, (val_images, val_labels) in enumerate(test_dataloader):
            # val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    total_row = 0
    total_num = 0
    for i in range(65):
        num = 0
        row = 0
        for j in range(65):
            num = confusion.matrix[i][j] + num
        # print('{}行 = '.format(i), num)
        for j in range(65):
            row = confusion.matrix[j][i] + row
        # print('{}列 = '.format(i), row)
        # total_row +=row
        # total_num += num

        for j in range(65):
            confusion.matrix[j][i] = confusion.matrix[j][i] / row
    # print(total_num, total_row)
    df = pd.DataFrame(confusion.matrix)
    # 将数据框保存为Excel文件
    df.to_excel('./Output/output.xlsx', index=False)
