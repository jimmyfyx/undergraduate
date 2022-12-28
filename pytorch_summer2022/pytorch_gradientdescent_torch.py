import torch
import torch.nn as nn
import numpy as np

## Simple Model y = w * x ##

'''
1) Design Model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
   - forward pass: compute prediction
   - backward pass: compute gradients
   - update parameters
'''

# Input dataset y = 2 * x #
X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

X_test = torch.tensor([5], dtype = torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

# # Use the pytorch built-in model "Linear"
# model = nn.Linear(input_size, output_size)

# Alternative way to build a model from scratch
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
num_itr = 100

loss = nn.MSELoss()                                                         # built-in function to calculate loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)         # built-in optimizer using gradient descent (SGD stand for stochastic gradient descent)

for i in range(num_itr):
    # use the built model to calculate predicted value
    output_predicted = model(X)

    # use built-in function to calculate loss
    total_loss = loss(Y, output_predicted)                                  

    # use autograd to calculate gradient dJ/dw
    total_loss.backward()

    # use optimizer to update parameters
    optimizer.step()
    
    # reinitialize gradients for the next iteration
    optimizer.zero_grad()

    if i % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {i + 1}: w = {w[0][0].item():.3f}, loss = {total_loss:.3f}')

print(f'Predicted after training: f(5) = {model(X_test).item():.3f}')