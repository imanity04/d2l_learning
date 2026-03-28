import os
import pandas as pd
import torch

data_file = os.path.join('.', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data)

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

"""
inputs.mean() 返回的类型是序列
NumRooms          3.00
RoofType_Slate    0.25
RoofType_nan      0.75
"""
inputs = inputs.fillna(inputs.mean())   # fillna对每一列进行操作，如果有na的就填入后面对应的值
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))

print(X)
print(y)