"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # initialize parameters
        n = X_train.shape[0]
        w_grad = np.zeros((self.w.shape[0], self.w.shape[1]))

        output = np.matmul(self.w, X_train.T)
        # loop through the score for each sample in the mini-batch
        for i in range(output.shape[1]):
            score = output[:, i]
            correct_label = y_train[i]
            correct_class_score = score[correct_label]
            for j in range(score.shape[0]):
                if j == correct_label:
                    continue
                else:
                    # the current class is not the correct class
                    if correct_class_score - score[j] < 1:
                        w_grad[correct_label] -= X_train[i]
                        w_grad[j] += X_train[i]
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
        
        if self.n_class == 2:
            # RICE dataset
            # first change the labels (0 to -1)
            y_train_rice = np.zeros(y_train.shape[0])
            for i in range(y_train.shape[0]):
                if y_train[i] == 0:
                    y_train_rice[i] = -1
                else:
                    y_train_rice[i] = 1
            self.w = np.random.rand(1, X_train.shape[1])

            # training for binary SVM
            for epoch in range(self.epochs):
                # loop through each sample in each epoch
                for i, x_i in enumerate(X_train):
                    prediction = np.matmul(self.w, X_train[i].T)
                    if y_train_rice[i] * prediction[0] >= 1:
                        self.w = self.w - self.lr * ((self.reg_const * self.w) / X_train.shape[0])
                    else:
                        self.w = self.w + self.lr * (y_train_rice[i] * X_train[i] - (self.reg_const * self.w) / X_train.shape[0])
                print('Epoch[{cur_epoch}/{total_epoch}]'.format(cur_epoch = epoch + 1, total_epoch = self.epochs))
            print('Training Finished!')
        else:
            # Multi-class SVM on Fashion-MNIST
            self.w = np.random.rand(self.n_class, X_train.shape[1]) 
            batch_size = 100 

            # training for multi-class SVM    
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

        if self.n_class == 2:
            output = np.reshape(output, output.shape[1])
            output[output >= 1] = 1
            output[output <= -1] = 0
            return output
        else:
            predict_labels = np.argmax(output, axis=0)
            return predict_labels
        
