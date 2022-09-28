from pickletools import float8
import torch
import numpy as np
import hw1_utils as utils


# Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        theta (2 x d Float Tensor): MAP estimation of P(X_j=1|Y=i)

    '''
    """
        X_train.shape = (100, 20)
        Y_train.shape = (100, 1)
    """
    # initializations
    N = X.size(dim = 0)
    d = X.size(dim = 1)
    theta = torch.zeros(2, d)
    prob_xd_y0 = torch.zeros(1, d)
    prob_xd_y1 = torch.zeros(1, d)

    # split dataset from y = 0 and 1
    data_y0 = X[y == 0]
    data_y1 = X[y == 1]

    # calculate P(X_j=1|Y=i)
    prob_xd_y0 = torch.sum(data_y0, 0) / (data_y0.size(dim=0))
    prob_xd_y1 = torch.sum(data_y1, 0) / (data_y1.size(dim=0))

    theta[0] = prob_xd_y0
    theta[1] = prob_xd_y1

    return theta


def bayes_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    return 1 - torch.sum(y) / y.size(dim = 0)


def bayes_classify(theta, p, X):
    '''
    Arguments:
        theta (2 x d Float Tensor): returned value of `bayes_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    # initializations
    N = X.size(dim = 0)
    d = X.size(dim = 1)
    y_label = torch.zeros(N)

    # for each object in the test set
    for i in range(N):
        # try y = 0 and y = 1 for each Xi for each object
        log_y0 = torch.zeros(d, 1)
        log_y1 = torch.zeros(d, 1)
        for j in range(d):
            if X[i][j] == 1:
                log_y0[j] = torch.log(theta[0][j])
                log_y1[j] = torch.log(theta[1][j])
            else:
                log_y0[j] = torch.log(1 - theta[0][j])
                log_y1[j] = torch.log(1 - theta[1][j])

        log_sum_y0 = torch.sum(log_y0)
        log_sum_y1 = torch.sum(log_y1)

        p_tensor = torch.tensor([p])
        p_compl_tensor = torch.tensor([1 - p])
        p_y0 = log_sum_y0 + torch.log(p_tensor)
        p_y1 = log_sum_y1 + torch.log(p_compl_tensor)

        # assign label for the current object
        if p_y0 > p_y1:
            y_label[i] = 0
        else:
            y_label[i] = 1

    return y_label



# Problem Gaussian Naive Bayes
def gaussian_MAP(X, y):
    '''
    Arguments:
        X (N x d FloatTensor): features of each object
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)

    '''
    # initializations
    N = X.size(dim = 0)
    d = X.size(dim = 1)
    mu = torch.zeros(2, d)
    sigma2 = torch.zeros(2, d)

    # split the dataset from y = 0 and 1
    X_y1 = X[y == 1]
    X_y0 = X[y == 0]

    mu[0] = torch.reshape(torch.mean(X_y0, 0), (1, d))
    mu[1] = torch.reshape(torch.mean(X_y1, 0), (1, d))
    sigma2[0] = torch.reshape(torch.var(X_y0, 0, unbiased=False), (1, d))
    sigma2[1] = torch.reshape(torch.var(X_y1, 0, unbiased=False), (1, d))

    return mu, sigma2


def gaussian_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)

    '''
    return 1 - torch.sum(y) / y.size(dim = 0)


def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x d Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x d Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1
    
    '''
    # initializations
    N = X.size(dim = 0)
    d = X.size(dim = 1)
    y_label = torch.zeros(N)
    one = torch.tensor([1])

    for i in range(N):
        log_y0 = torch.zeros(d)
        log_y1 = torch.zeros(d)
        
        log_y0 = torch.log((1 / torch.sqrt(2 * np.pi * sigma2[0]))) + ((-0.5 * (X[i] - mu[0])**2) / sigma2[0]) * torch.log(torch.exp(one))
        log_y1 = torch.log((1 / torch.sqrt(2 * np.pi * sigma2[1]))) + ((-0.5 * (X[i] - mu[1])**2) / sigma2[1]) * torch.log(torch.exp(one))

        p_tensor = torch.tensor([p])
        p_compl_tensor = torch.tensor([1 - p])
        prob_y0 = torch.log(p_tensor) + torch.sum(log_y0)
        prob_y1 = torch.log(p_compl_tensor) + torch.sum(log_y1)

        if prob_y0 > prob_y1:
            y_label[i] = 0
        else:
            y_label[i] = 1
    
    return y_label



