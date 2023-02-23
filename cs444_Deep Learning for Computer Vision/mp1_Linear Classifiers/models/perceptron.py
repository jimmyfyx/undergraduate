"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.random.rand(self.n_class, X_train.shape[1])  # shape (10, 784)
        lr_decay_rate = 1.5

        for epoch in range(self.epochs):
            # loop through each sample in each epoch
            for i, x_i in enumerate(X_train):
                # calculate prediction
                x_i = np.reshape(x_i, (x_i.shape[0], 1))
                output = np.matmul(self.w, x_i)  # shape (10, 1)

                # predicted value with the weight of correct label
                correct_label = y_train[i]
                score_correct = output[correct_label]
                
                # update weights
                for j in range(output.shape[0]):
                    if j != correct_label and output[j][0] > score_correct:
                        self.w[j] = self.w[j] - self.lr * x_i.T
                        self.w[correct_label] = self.w[correct_label] + self.lr * x_i.T
            
            # update new learning rate
            self.lr = (1 / (1 + lr_decay_rate * (epoch + 1))) * self.lr

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
        predict_labels = np.argmax(output, axis=0)
        return predict_labels
        
        
