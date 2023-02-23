"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # initializations
        self.w = np.random.rand(1, X_train.shape[1])
        # first change the labels (0 to -1)
        y_train_rice = np.zeros(y_train.shape[0])
        for i in range(y_train.shape[0]):
            if y_train[i] == 0:
                y_train_rice[i] = -1
            else:
                y_train_rice[i] = 1
        
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                w_grad = self.sigmoid(np.matmul(self.w, X_train[i].T) * (-y_train_rice[i])) * X_train[i] * y_train_rice[i].T
                self.w = self.w + self.lr * w_grad

            print('Epoch[{cur_epoch}/{total_epoch}]'.format(cur_epoch = epoch + 1, total_epoch = self.epochs))
        print('Training Finished!')

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        output = np.matmul(self.w, X_test.T)
        output = self.sigmoid(output)
        output[output < self.threshold] = 0
        output[output >= self.threshold] = 1
        output = np.reshape(output, output.shape[1]).astype(int)
        return output
