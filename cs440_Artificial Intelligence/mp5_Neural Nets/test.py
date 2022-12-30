import torch
import numpy as np

input = torch.randn(1, 2883)
input = input.view(1, 3, 31, 31)

conv1 = torch.nn.Conv2d(3, 16, 3, padding=0, stride=2)
input = conv1(input)
batch_1 = torch.nn.BatchNorm2d(16)
input = batch_1(input)
# pool_1 = torch.nn.MaxPool2d(2, 2)
# input = pool_1(input)


conv2 = torch.nn.Conv2d(16, 32, 3, padding=0, stride=2)
input = conv2(input)
batch_2 = torch.nn.BatchNorm2d(32)
input = batch_2(input)
# pool_2 = torch.nn.MaxPool2d(2, 2)
# input = pool_2(input)


conv3 = torch.nn.Conv2d(32, 64, 3, padding=0, stride=2)
input = conv3(input)
batch_3 = torch.nn.BatchNorm2d(64)
input = batch_3(input)
# pool_3 = torch.nn.MaxPool2d(2, 2)
# input = pool_3(input)

print(input.shape)
1/0

conv4 = torch.nn.Conv2d(256, 256, 3, padding=1)
input = conv4(input)
pool_2 = torch.nn.MaxPool2d(2, 2)
input = pool_2(input)
flatten = torch.nn.Flatten(1, 3)
input = flatten(input)
print(input.shape)
linear_1 = torch.nn.Linear(256 * 7 * 7, 1024)
input = linear_1(input)
dropout_1 = torch.nn.Dropout(p=0.5)
input = dropout_1(input)
print(input.shape)

a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
print(a)
mean = torch.sum(a, dim=0) / 3
var = torch.var(a, dim=0)
print(var.shape)
a = (a - mean) / np.sqrt(var)
print(a)