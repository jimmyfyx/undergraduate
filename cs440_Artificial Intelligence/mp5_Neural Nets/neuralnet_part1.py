# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.hidden_size = 100
        self.lr = lrate
        self.model = nn.Sequential(
                        nn.Linear(in_size, self.hidden_size),
                        nn.ReLU(),
                        nn.Linear(self.hidden_size, out_size)
                    )
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        return self.model(x)


    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        self.optimizer.zero_grad()

        # calculate loss
        batch_size = x.shape[0]
        outputs = self(x)

        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().numpy())



def fit(train_set,train_labels,dev_set,epochs,batch_size=50):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    # initialize the neural net
    loss_function = nn.CrossEntropyLoss()
    net = NeuralNet(lrate=0.001, loss_fn=loss_function, in_size=train_set.shape[1], out_size=4)

    # normalize data (train and dev)
    dataset_mean = torch.sum(train_set, dim=0) / train_set.shape[0]
    dataset_var = torch.var(train_set, dim=0)
    train_set = (train_set - dataset_mean) / np.sqrt(dataset_var)
    dev_set = (dev_set - dataset_mean) / np.sqrt(dataset_var)

    # load data
    train_dataset = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    losses = []
    # training step
    for epoch in range(epochs):
        epoch_loss = 0
        num_batch = 0
        for batch in train_loader:
            num_batch += 1
            batch_loss = net.step(batch['features'], batch['labels'])
            
            epoch_loss += batch_loss
        losses.append(epoch_loss / num_batch)

    # evaluation step
    pred_labels = np.empty([dev_set.shape[0]])
    y_hat = net(dev_set).detach().cpu().numpy()
    for i in range(pred_labels.shape[0]):
        pred_labels[i] = np.argmax(y_hat[i])
    
    return losses,pred_labels.astype(int),net
