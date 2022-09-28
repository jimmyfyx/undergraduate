import torch
import hw2_utils as utils
import matplotlib.pyplot as plt


'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # initializations
    n = X.size(dim = 0)
    d = X.size(dim = 1)
    weights = torch.zeros(d + 1, 1)
    prepend_ones = torch.ones(1, n)
    X_t_prepend = torch.cat((prepend_ones, torch.t(X)), 0)

    # number of iterations
    for i in range(num_iter):
        # calculate the loss
        gradient = torch.matmul(torch.matmul(torch.t(weights), X_t_prepend) - torch.t(Y), torch.t(X_t_prepend)) * (2 / n)
        weights = weights - torch.reshape(gradient, (d + 1, 1)) * lrate
    
    return weights


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # parameters
    n = X.size(dim = 0)
    d = X.size(dim = 1)

    # prepend a column of ones to X
    prepend_ones = torch.ones(n, 1)
    X_ = torch.cat((prepend_ones, X), 1)

    # solve for weights
    weights = torch.matmul(torch.linalg.pinv(X_), Y)

    return weights


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    # obtain data
    X_train, Y_train = utils.load_reg_data()
    weights = linear_normal(X_train, Y_train)

    # process data
    torch.flatten(X_train)
    torch.flatten(Y_train)

    # plot
    plt.scatter(X_train, Y_train)
    plt.plot(X_train, weights[1] * X_train + weights[0])
    
    return plt.gcf()



# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    # initializations
    n = X.size(dim = 0)
    d = X.size(dim = 1)
    weights = torch.zeros(d + 1, 1)
    prepend_ones = torch.ones(n, 1)
    X_prepend = torch.cat((prepend_ones, X), 1)

    for i in range(num_iter):
        gradient = (-Y * X_prepend) / (torch.exp(Y * torch.matmul(X_prepend, weights)) + 1) * (1 / n)
        gradient = torch.sum(gradient, 0)
        gradient = torch.reshape(gradient, (d+1, 1))
        weights = weights - gradient * lrate 

    return weights



def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    
    # obtain data
    # X_train is a 2-d input, Y_train has is 1-d
    X_train, Y_train = utils.load_logistic_data()
    weights_log = logistic(X_train, Y_train)
    weights_lin = linear_normal(X_train, Y_train)

    # process data
    X1 = X_train[:, 0:1]
    X2 = X_train[:, 1:]
    torch.flatten(X1)
    torch.flatten(X2)

    # plot
    plt.title('Logistic Regression v.s. Linear Regression')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.scatter(X1, X2)
    plt.plot(X1, -(weights_log[0] + weights_log[1] * X1) / weights_log[2])
    plt.plot(X1, -(weights_lin[0] + weights_lin[1] * X1) / weights_lin[2], color='red')
    
    return plt.gcf()

"""
X_train, Y_train = utils.load_reg_data()
print(X_train.size())
print(Y_train.size())
# print(X_train)
weights = logistic(X_train, Y_train)
print(weights)
"""

"""
X_train = torch.tensor([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]])
Y_train = torch.tensor([[2], [4], [6], [8]])
weights = linear_gd(X_train, Y_train, num_iter=3000)
print(weights.size())
print(weights)
"""
# logistic_vs_ols()