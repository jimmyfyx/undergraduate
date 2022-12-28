from ossaudiodev import SOUND_MIXER_ALTPCM
import hw3_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    # initializations
    N = x_train.shape[0]
    d = x_train.shape[1]
    alpha = torch.zeros((N, 1), requires_grad=True)

    for i in range(num_iters):
        # calculate loss function
        loss_function = 0
        alpha_sum = 0
        for i in range(N):
            for j in range(N):
                loss_function += alpha[i] * alpha[j] * y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])
            alpha_sum = alpha_sum + alpha[i]
        loss_function = -0.5 * loss_function + alpha_sum

        if alpha.grad:
            alpha.grad.zero_()

        # calculate gradient
        loss_function.backward()

        # udpate alpha (gradient ascent)
        with torch.no_grad():
            alpha += lr * alpha.grad
            alpha = torch.clamp(alpha, min=0, max=c)
        alpha.requires_grad_()
    
    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    N = x_train.shape[0]
    M = x_test.shape[0]

    # find the minimum alpha index
    min_alpha_idx = 0
    min_alpha = float('inf')
    for i in range(N):
        if alpha[i] < min_alpha and alpha[i] != 0:
            min_alpha = alpha[i]
            min_alpha_idx = i

    # calculate wx
    weights_x = 0
    for i in range(N):
        weights_x = weights_x + alpha[i] * y_train[i] * kernel(x_train[i], x_train[min_alpha_idx])

    # calculate b
    b = (-y_train[min_alpha_idx] * weights_x + 1) / y_train[min_alpha_idx]

    classifier = torch.zeros(M)
    for i in range(M):
        sum = 0
        for j in range(N):
            sum += alpha[j] * y_train[j] * kernel(x_train[j], x_test[i])
        sum = sum + b
        classifier[i] = sum

    print(classifier.shape)
    return classifier


class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=7, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(7)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=2)
        self.batchnorm2 = nn.BatchNorm2d(3)
        self.linear = nn.Linear(3 * 2 * 2, 10)


    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        xb = torch.unsqueeze(xb, 1)
        xb = self.conv1(xb)
        xb = F.relu(self.batchnorm1(xb))
        xb = self.maxpool(xb)
        xb = self.conv2(xb)
        xb = F.relu(self.batchnorm2(xb))
        xb = torch.reshape(xb, (xb.shape[0], xb.shape[1] * xb.shape[2] * xb.shape[3]))
        xb = self.linear(xb)
        return xb


def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss
    net.eval()
    with torch.no_grad():
        train_losses.append(hw3_utils.epoch_loss(net, loss_func, train_dl))
        test_losses.append(hw3_utils.epoch_loss(net, loss_func, test_dl))

    for epoch in range(n_epochs):
        # each epoch
        net.train()
        for i, (images, labels) in enumerate(train_dl):
            # each batch
            hw3_utils.train_batch(net, loss_func, images, labels, optimizer)

        net.eval()
        # calculate epoch losses
        with torch.no_grad():
            train_losses.append(hw3_utils.epoch_loss(net, loss_func, train_dl))
            test_losses.append(hw3_utils.epoch_loss(net, loss_func, test_dl))

    return train_losses, test_losses

"""
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[3], [3], [3]])
c = torch.tensor([1, 2, 3, 4])
print(a)
print(b)
# a = torch.unsqueeze(a, 1)
# print(a.shape)
print(a * b)
print(torch.sum(a * b, 0))
"""

"""
model = DigitsConvNet()
train, test = hw3_utils.torch_digits()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
train_losses, test_losses = fit_and_evaluate(net=model, optimizer=optimizer, loss_func=loss_function, train=train, test=test, n_epochs=30, batch_size=8)
epoch = np.linspace(0, 30, 31)
plt.title('Training Loss vs. Test Loss')
plt.xlabel('num_epoch')
plt.ylabel('loss')
plt.plot(epoch, train_losses, label='train_losses')
plt.plot(epoch, test_losses, color='orange', label='test_losses')
plt.legend()
plt.show()
"""

"""
x_train, y_train = hw3_utils.xor_data()
alpha = svm_solver(x_train, y_train, 0.1, 10000, hw3_utils.rbf(5))

hw3_utils.svm_contour(lambda x_test: svm_predictor(alpha, x_train, y_train, x_test, hw3_utils.rbf(5)), xmin=-8, xmax=8, ymin=-8, ymax=8, ngrid = 33)
"""