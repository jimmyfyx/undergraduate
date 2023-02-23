"""Neural network model."""

from typing import Sequence

import numpy as np
from copy import deepcopy


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
        
        # parameters related to Adam
        self.t = 0
        self.v_dw = {}
        self.s_dw = {}
        self.v_db = {}
        self.s_db = {}
        for para_name in self.params.keys():
            if para_name[0] == 'W':
                self.v_dw[para_name] = 0
                self.s_dw[para_name] = 0
            else:
                self.v_db[para_name] = 0
                self.s_db[para_name] = 0
       

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        return np.matmul(X, W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data (N, hidden_size)
        Returns:
            the output data
        """
        relu_grad = deepcopy(X)
        relu_grad[relu_grad > 0] = 1
        relu_grad[relu_grad <= 0] = 0
        return relu_grad

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # y and p both have shape (N, num_classes)
        loss = np.sum(np.sum((y - p) * (y - p), axis=1))
        return loss * (1 / y.shape[0])

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        self.outputs["Layer" + str(0)] = X
        self.outputs["Layer" + str(0) + "_relu"] = X
    
        for i in range(1, self.num_layers + 1):
            # calculate the output of each layer
            output = self.linear(self.params["W" + str(i)], self.outputs["Layer" + str(i - 1) + "_relu"], self.params["b" + str(i)])
            self.outputs["Layer" + str(i)] = output

            # not applying ReLU activation for the last layer
            if i < self.num_layers:
                output = self.relu(output)
                self.outputs["Layer" + str(i) + "_relu"] = output
            
        # apply sigmoid to the last layer
        self.outputs["OUT"] = self.sigmoid(self.outputs["Layer" + str(self.num_layers)])
        return self.outputs["OUT"]

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets (N, num_classes)
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        xi_gradients = {}

        # calculate the MSE loss and MSE gradient w.r.t. predicted output
        mse_loss = self.mse(y, self.outputs["OUT"])
        mse_grad = -2 * (y - self.outputs["OUT"]) / y.shape[0]
        
        # gradient through sigmoid
        sigmoid_grad_local = self.sigmoid(self.outputs["Layer" + str(self.num_layers)]) * (1 - self.sigmoid(self.outputs["Layer" + str(self.num_layers)]))
        before_sigmoid_grad = sigmoid_grad_local * mse_grad

        xi_gradients["X" + str(self.num_layers)] = before_sigmoid_grad
        for i in range(self.num_layers, 0, -1):
            # calculate gradients for layer i parameters
            wi_grad = np.matmul(self.outputs["Layer" + str(i - 1) + "_relu"].T, xi_gradients["X" + str(i)])
            bi_grad = np.sum(xi_gradients["X" + str(i)], axis=0)
            self.gradients["W" + str(i)] = wi_grad
            self.gradients["b" + str(i)] = bi_grad

            # calculate the gradient for output from previous layer
            xi_grad = np.matmul(xi_gradients["X" + str(i)], self.params["W" + str(i)].T) * self.relu_grad(self.outputs["Layer" + str(i - 1)])
            xi_gradients["X" + str(i - 1)] = xi_grad

        return mse_loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        if opt == "SGD":
            '''
            Normal SGD
            '''
            for para_name in self.params.keys():
                self.params[para_name] = self.params[para_name] - lr * self.gradients[para_name]
        else:
            '''
            Adam Optimizer
            '''
            self.t += 1
            # update v_dw, v_db, s_dw, s_db
            for para_name in self.params.keys():
                if para_name[0] == 'W':
                    self.v_dw[para_name] = b1 * self.v_dw[para_name] + (1 - b1) * self.gradients[para_name]
                    self.s_dw[para_name] = b2 * self.s_dw[para_name] + (1 - b2) * (self.gradients[para_name] ** 2)
                else:
                    self.v_db[para_name] = b1 * self.v_db[para_name] + (1 - b1) * self.gradients[para_name]
                    self.s_db[para_name] = b2 * self.s_db[para_name] + (1 - b2) * self.gradients[para_name]

            # correct v_dw, v_db, s_dw, s_db
            # v_dw_corr = self.v_dw / (1 - b1 ** self.t)
            # v_db_corr = self.v_db / (1 - b1 ** self.t)
            # s_dw_corr = self.s_dw / (1 - b2 ** self.t)
            # s_db_corr = self.s_db / (1 - b2 ** self.t)

            # update weights
            for para_name in self.params.keys():
                if para_name[0] == 'W':
                    self.params[para_name] = self.params[para_name] - lr * ((self.v_dw[para_name] / (1 - np.power(b1, self.t))) / (np.sqrt(self.s_dw[para_name] / (1 - np.power(b2, self.t))) + eps))
                else:
                    self.params[para_name] = self.params[para_name] - lr * ((self.v_db[para_name] / (1 - np.power(b1, self.t))) / (np.sqrt(self.s_db[para_name] / (1 - np.power(b2, self.t))) + eps))
