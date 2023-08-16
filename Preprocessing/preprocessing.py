import os
import glob
import pandas as pd
import os
import torch
import torch.nn as nn
# import einops
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from myDataset import myDataset

# in_dir = r"train_csv"
# 获取文件夹中所有CSV文件的文件名
csv_files = glob.glob('./train_xd_csv/*.csv')
# print(csv_files) 

# 遍历所有CSV文件并对它们进行分类
for file in csv_files:
    # 读取CSV文件
    data = pd.read_csv(file, nrows = 400)
    
    # # 删除特征
    # # 删除指定的列
    # columns_to_drop = ['Time', 'Hips.X', 'Hips.Y', 'Hips.Z', 'Spine1.X', 'Spine1.Y', 'Spine1.Z', 'Spine2.X', 'Spine2.Y', 'Spine2.Z', 'Spine3.X', 'Spine3.Y', 'Spine3.Z', 'Spine4.X', 'Spine4.Y', 'Spine4.Z', 'Neck.X', 'Neck.Y', 'Neck.Z', 'Head.X', 'Head.Y', 'Head.Z', 'HeadEnd.X', 'HeadEnd.Y', 'HeadEnd.Z', 'RightThigh.X', 'RightThigh.Y', 'RightThigh.Z', 'RightShin.X', 'RightShin.Y', 'RightShin.Z', 'RightFoot.X', 'RightFoot.Y', 'RightFoot.Z', 'RightToe.X', 'RightToe.Y', 'RightToe.Z', 'RightToeEnd.X', 'RightToeEnd.Y', 'RightToeEnd.Z', 'LeftThigh.X', 'LeftThigh.Y', 'LeftThigh.Z', 'LeftShin.X', 'LeftShin.Y', 'LeftShin.Z', 'LeftFoot.X', 'LeftFoot.Y', 'LeftFoot.Z', 'LeftToe.X', 'LeftToe.Y', 'LeftToe.Z', 'LeftToeEnd.X', 'LeftToeEnd.Y', 'LeftToeEnd.Z']  # 要删除的列的名称列表
    # data.drop(columns=columns_to_drop, inplace=True)
    # # 数据补全
    # # 指定要扩展到的行数
    # target_rows = 400

    # # 计算需要添加的行数
    # add_rows = target_rows - data.shape[0]

    # # 创建新行，用0来填充
    # new_row = pd.DataFrame({'RightShoulder.X': [0]})

    # if add_rows > 0:
    #     # 循环添加新行到CSV文件中
    #     for i in range(add_rows):
    #         data = pd.concat([data, new_row], ignore_index=True, sort=False)
    #         # data = data.append(new_row, ignore_index=True)
    #         num = len(data.columns) - 1
    #         for i in range(num):
    #             data.iloc[-1, -(i+1)] = 0

    data = data.values.tolist()
    # print(len(data))
    data = np.array(data)
    data = data[:, 1:]
    # print(len(data[0]))
    temp = np.zeros((169), dtype=np.float32)
    # print(temp)
    x = 0
    for i in range(len(data)):
        # print(data[0][0])
        if i != 0 and data[i][1]!=0 and data[i][2]!=0 and data[i][43]!=0:
            # print(len(data[i]))
            for j in range(len(data[i])):
                temp[j] = data[i][j] - data[i - 1][j]
                if i != 1:
                    data[i - 1][j] = x
                x = temp[j]

            # data[i][1:] = data[i][1:] - data[i-1][1:]
    # print(data)
    for i in range(400):
        for j in range(len(data[i])):
            data[i][j] = data[i][j] * 1000
    print(file)
    # # 反转数据以恢复原始顺序
    # data = reversed(data)
    # print(file)
    # data.to_csv(file)
    
    pd.DataFrame(data).to_csv(file, index = True, header = True)