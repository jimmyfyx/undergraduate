import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



def sigmoid(a):
    return 1 / (1 + torch.exp(-a))

def softmax(b):
    deno = 0
    for i in range(b.shape[0]):
        deno += torch.exp(b[i])

    return torch.exp(b) / deno

def loss_function(y_hat, y):
    sum = 0
    for i in range(3):
        sum += y[i] * torch.log(y_hat[i])

    return -sum


x = torch.tensor([[1], [1], [1], [0], [0], [1], [1]], dtype=torch.float64)
y = torch.tensor([[0], [1], [0]], dtype=torch.float64)
alpha = torch.tensor([[1, 1, 2, 3, 0, 1, 3], [1, 3, 1, 2, 1, 0, 2], [1, 2, 2, 2, 2, 2, 1], [1, 1, 0, 2, 1, 2, 2]], dtype=torch.float64)
beta = torch.tensor([[1, 1, 2, 2, 1], [1, 1, 1, 1, 2], [1, 3, 1, 1, 1]], dtype=torch.float64)
one = torch.tensor([[1.0]], dtype=torch.float64)

a = torch.matmul(alpha, x)
print(a)
z = sigmoid(a)
print("z = {z_matrix}".format(z_matrix = z))
z = torch.cat((one, z), dim=0)
b = torch.matmul(beta, z)
print(b)
y_hat = softmax(b)
print("y_hat = {y_h}".format(y_h = y_hat))
loss = loss_function(y_hat, y)
# print(loss)
# print(y_hat)
print(-(1 - y_hat[0]))