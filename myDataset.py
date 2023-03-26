import os
import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import glob


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)

        img = img[np.newaxis, :]
        # print(img.shape)

        c, h, w = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(c, h, w))
        N = np.repeat(N, c, axis=2)
        # print(N)

        img = N + img
        # print(img)
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img


class myDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        # """
        # data_dir: 数据文件路径
        # """"
        # 读文件夹下每个数据文件的名称
        self.file_name = os.listdir(data_dir)

        self.data_path = glob.glob('{}/*.csv'.format(data_dir))
        print(len(self.data_path))
        # 把每一个文件的路径拼接起来
        # for index in range(len(self.file_name)):
        #     if not self.file_name[index].startswith('.'):
        #         self.data_path.append(os.path.join(data_dir, self.file_name[index]))
        # print(self.data_path)
        # self.data = pd.read_csv(self.data_path[1])
        # print(self.data)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        # 读取每一个数据
        # print(self.data_path[index])
        index = index - 1
        # print(self.data_path[index])
        column = [x for x in range(169)]
        column = column[12:169]
        del column[74: 85]
        # print(len(column))
        data = pd.read_csv(self.data_path[index], nrows=400, encoding_errors="replace", engine='python', usecols=column)

        # # 删除特征 # 删除指定的列 columns_to_drop = ['Time', 'Hips.X', 'Hips.Y', 'Hips.Z', 'Spine1.X', 'Spine1.Y',
        # 'Spine1.Z', 'Spine2.X', 'Spine2.Y', 'Spine2.Z', 'Spine3.X', 'Spine3.Y', 'Spine3.Z', 'Spine4.X', 'Spine4.Y',
        # 'Spine4.Z', 'Neck.X', 'Neck.Y', 'Neck.Z', 'Head.X', 'Head.Y', 'Head.Z', 'HeadEnd.X', 'HeadEnd.Y',
        # 'HeadEnd.Z', 'RightThigh.X', 'RightThigh.Y', 'RightThigh.Z', 'RightShin.X', 'RightShin.Y', 'RightShin.Z',
        # 'RightFoot.X', 'RightFoot.Y', 'RightFoot.Z', 'RightToe.X', 'RightToe.Y', 'RightToe.Z', 'RightToeEnd.X',
        # 'RightToeEnd.Y', 'RightToeEnd.Z', 'LeftThigh.X', 'LeftThigh.Y', 'LeftThigh.Z', 'LeftShin.X', 'LeftShin.Y',
        # 'LeftShin.Z', 'LeftFoot.X', 'LeftFoot.Y', 'LeftFoot.Z', 'LeftToe.X', 'LeftToe.Y', 'LeftToe.Z',
        # 'LeftToeEnd.X', 'LeftToeEnd.Y', 'LeftToeEnd.Z']  # 要删除的列的名称列表 data.drop(columns=columns_to_drop,
        # inplace=True) # 数据补全 # 指定要扩展到的行数 target_rows = 400
        #
        # # 计算需要添加的行数
        # add_rows = target_rows - data.shape[0]
        #
        # # 创建新行，用0来填充
        # new_row = pd.DataFrame({'RightShoulder.X': [0]})
        #
        # if add_rows > 0:
        #     # 循环添加新行到CSV文件中
        #     for i in range(add_rows):
        #         data = pd.concat([data, new_row], ignore_index=True, sort=False)
        #         # data = data.append(new_row, ignore_index=True)
        #         num = len(data.columns) - 1
        #         for i in range(num):
        #             data.iloc[-1, -(i+1)] = 0
        transform = AddGaussianNoise(mean=0, variance=1, amplitude=20)
        data = transform(data)
        # 转成张量
        data = torch.tensor(data, dtype=torch.float32)
        # print(data.size())
        data = data.view(1, 400, 146)
        # print(data.size())
        # 数据归一化
        trans = transforms.Normalize(mean=[0.5], std=[0.5])
        data = trans(data)
        # data = data.view(400, 169)

        # data = data.type(torch.LongTensor
        # data = torch.FloatTensor(data.values)
        # data = data[:400, :168]

        # 读取每个数据对应的label
        str = os.path.basename(self.data_path[index])
        label = str[3:6]
        d = {'111': 0, '112': 1, '113': 2, '121': 3, '122': 4,
             '123': 5, '131': 6, '132': 7, '133': 8, '141': 9,
             '142': 10, '143': 11, '151': 12, '152': 13, '153': 14,
             '211': 15, '212': 16, '213': 17, '221': 18, '222': 19,
             '223': 20, '311': 21, '312': 22, '313': 23, '321': 24,
             '322': 25, '323': 26, '331': 27, '332': 28, '333': 29,
             '341': 30, '342': 31, '343': 32, '411': 33, '412': 34,
             '413': 35, '421': 36, '422': 37, '431': 38, '432': 39,
             '433': 40, '441': 41, '442': 42, '443': 43, '451': 44,
             '452': 45, '453': 46, '511': 47, '512': 48, '521': 49,
             '522': 50, '523': 51, '531': 52, '532': 53, '611': 54,
             '612': 55, '621': 56, '622': 57, '631': 58, '632': 59,
             '633': 60, '641': 61, '642': 62, '651': 63, '652': 64,
             }
        label = d[label]
        # print(label)
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.float32)
        # label = torch.FloatTensor(label)

        return data, label


class testDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        # """
        # data_dir: 数据文件路径
        # """"
        # 读文件夹下每个数据文件的名称
        self.file_name = os.listdir(data_dir)

        self.data_path = glob.glob('{}/*.csv'.format(data_dir))
        # print(len(self.data_path))
        # 把每一个文件的路径拼接起来
        # for index in range(len(self.file_name)):
        #     if not self.file_name[index].startswith('.'):
        #         self.data_path.append(os.path.join(data_dir, self.file_name[index]))
        # print(self.data_path)
        # self.data = pd.read_csv(self.data_path[1])
        # print(self.data)

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        # 读取每一个数据
        # print(self.data_path[index])
        index = index - 1
        # print(self.data_path[index])
        column = [x for x in range(169)]
        column = column[12:169]
        del column[74: 85]
        # print(len(column))
        data = pd.read_csv(self.data_path[index], nrows=400, encoding_errors="replace", engine='python',
                           usecols=column)

        # # 删除特征 # 删除指定的列 columns_to_drop = ['Time', 'Hips.X', 'Hips.Y', 'Hips.Z', 'Spine1.X', 'Spine1.Y',
        # 'Spine1.Z', 'Spine2.X', 'Spine2.Y', 'Spine2.Z', 'Spine3.X', 'Spine3.Y', 'Spine3.Z', 'Spine4.X', 'Spine4.Y',
        # 'Spine4.Z', 'Neck.X', 'Neck.Y', 'Neck.Z', 'Head.X', 'Head.Y', 'Head.Z', 'HeadEnd.X', 'HeadEnd.Y',
        # 'HeadEnd.Z', 'RightThigh.X', 'RightThigh.Y', 'RightThigh.Z', 'RightShin.X', 'RightShin.Y', 'RightShin.Z',
        # 'RightFoot.X', 'RightFoot.Y', 'RightFoot.Z', 'RightToe.X', 'RightToe.Y', 'RightToe.Z', 'RightToeEnd.X',
        # 'RightToeEnd.Y', 'RightToeEnd.Z', 'LeftThigh.X', 'LeftThigh.Y', 'LeftThigh.Z', 'LeftShin.X', 'LeftShin.Y',
        # 'LeftShin.Z', 'LeftFoot.X', 'LeftFoot.Y', 'LeftFoot.Z', 'LeftToe.X', 'LeftToe.Y', 'LeftToe.Z',
        # 'LeftToeEnd.X', 'LeftToeEnd.Y', 'LeftToeEnd.Z']  # 要删除的列的名称列表 data.drop(columns=columns_to_drop,
        # inplace=True) # 数据补全 # 指定要扩展到的行数 target_rows = 400
        #
        # # 计算需要添加的行数
        # add_rows = target_rows - data.shape[0]
        #
        # # 创建新行，用0来填充
        # new_row = pd.DataFrame({'RightShoulder.X': [0]})
        #
        # if add_rows > 0:
        #     # 循环添加新行到CSV文件中
        #     for i in range(add_rows):
        #         data = pd.concat([data, new_row], ignore_index=True, sort=False)
        #         # data = data.append(new_row, ignore_index=True)
        #         num = len(data.columns) - 1
        #         for i in range(num):
        #             data.iloc[-1, -(i+1)] = 0
        # transform = AddGaussianNoise(mean=0, variance=1, amplitude=20)
        # data = transform(data)
        # 转成张量
        data = torch.tensor(data.values, dtype=torch.float32)
        # print(data.size())
        data = data.view(1, 400, 146)
        # print(data.size())
        # 数据归一化
        trans = transforms.Normalize(mean=[0.5], std=[0.5])
        data = trans(data)
        # data = data.view(400, 169)

        # data = data.type(torch.LongTensor
        # data = torch.FloatTensor(data.values)
        # data = data[:400, :168]

        # 读取每个数据对应的label
        str = os.path.basename(self.data_path[index])
        label = str[3:6]
        d = {'111': 0, '112': 1, '113': 2, '121': 3, '122': 4,
             '123': 5, '131': 6, '132': 7, '133': 8, '141': 9,
             '142': 10, '143': 11, '151': 12, '152': 13, '153': 14,
             '211': 15, '212': 16, '213': 17, '221': 18, '222': 19,
             '223': 20, '311': 21, '312': 22, '313': 23, '321': 24,
             '322': 25, '323': 26, '331': 27, '332': 28, '333': 29,
             '341': 30, '342': 31, '343': 32, '411': 33, '412': 34,
             '413': 35, '421': 36, '422': 37, '431': 38, '432': 39,
             '433': 40, '441': 41, '442': 42, '443': 43, '451': 44,
             '452': 45, '453': 46, '511': 47, '512': 48, '521': 49,
             '522': 50, '523': 51, '531': 52, '532': 53, '611': 54,
             '612': 55, '621': 56, '622': 57, '631': 58, '632': 59,
             '633': 60, '641': 61, '642': 62, '651': 63, '652': 64,
             }
        label = d[label]
        # print(label)
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.float32)
        # label = torch.FloatTensor(label)

        return data, label
