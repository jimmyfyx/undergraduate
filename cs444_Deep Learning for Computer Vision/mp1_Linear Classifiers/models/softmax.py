"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # initialize parameters
        w_grad = np.zeros((self.w.shape[0], self.w.shape[1]))   # (10, 784)
        n = X_train.shape[0]

        output = np.matmul(self.w, X_train.T)    # (10, 100)
        # loop through each sample 
        for i in range(n):
            scores = output[:, i]
            scores = scores - np.max(scores)  # prevent overflow
            exp_sum = np.sum(np.exp(scores))
            correct_label = y_train[i]
            # loop through each class of a sample
            for j in range(scores.shape[0]):
                score_exp = np.exp(scores[j])
                if j == correct_label:
                    w_grad[correct_label] += ((score_exp / exp_sum) - 1) * X_train[i]
                else: 
                    w_grad[j] += (score_exp / exp_sum) * X_train[i]

        # average the gradients over mini-batch
        w_grad = w_grad / n
        # apply regularization
        w_grad = w_grad + self.w * self.reg_const 
        return w_grad 

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # initializations
        self.w = np.random.rand(self.n_class, X_train.shape[1]) 
        batch_size = 16
        lr_decay_rate = 1.0

        for epoch in range(self.epochs): 
            for batch_idx in range(int(X_train.shape[0] / batch_size)):
                # randomly create a mini-batch
                batch = np.random.choice(X_train.shape[0], batch_size)
                X_batch = X_train[batch]
                Y_batch = y_train[batch]

                # calculate w_grad for this batch
                w_grad = self.calc_gradient(X_batch, Y_batch)
                # update w
                self.w = self.w - self.lr * w_grad
        
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
