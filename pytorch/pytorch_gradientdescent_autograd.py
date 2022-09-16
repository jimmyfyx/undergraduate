import torch
import numpy as np

## Simple Model y = w * x ##

# Input dataset y = 2 * x #
X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

# Initialize parameters
w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model prediction
def forward(x):
    return w * x

# loss function
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
num_itr = 100

for i in range(num_itr):
    output_predicted = forward(X)
    total_loss = loss(Y, output_predicted)

    # use autograd to calculate gradient dJ/dw
    total_loss.backward()

    with torch.no_grad():
        # update parameters
        w.sub_(w.grad * learning_rate)
    
        # reinitialize gradients for the next iteration
        w.grad.zero_()

    if i % 10 == 0:
        print(f'epoch {i + 1}: w = {w:.3f}, loss = {total_loss:.3f}')

print(f'Predicted after training: f(5) = {forward(5):.3f}')