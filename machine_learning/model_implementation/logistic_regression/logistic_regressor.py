import numpy as np
from copy import deepcopy
from logistic_regression_parameters import LogisticRegressionParameters
from machine_learning.model_implementation.base_classes.supervised_model import SupervisedModel


class LogisticRegressor(SupervisedModel):
    def __init__(self, parameters: LogisticRegressionParameters):
        self.__parameters = deepcopy(parameters)
        self.__ws = deepcopy(self.__parameters.initial_weights)
        self.__b = deepcopy(self.__parameters.initial_bias)
        self.__train_loss = []

    def fit(self, X_train, y_train, print_loss = False):
        for _ in range(0, self.__parameters.epochs):
            self.__sgd_update(X_train, y_train)
            loss = self.loss(X_train, y_train)
            self.__train_loss.append(loss)
            if print_loss:
                print(f'loss = {loss}')

    def loss(self, X, y):
        predictions = self.predict(X)
        fst_term = y * np.log(predictions)
        sec_term = (1 - y) * np.log(1 - predictions)
        return -np.mean(fst_term + sec_term)

    def predict(self, X_pred):
        linear_predictions = X_pred.mul(self.__ws).sum(axis = 1) + self.__b
        return np.array(LogisticRegressor.sigmoid(linear_predictions))

    def get_weights(self):
        return self.__ws
    
    def get_bias(self):
        return self.__b

    def get_train_loss(self):
        return self.__train_loss

    def get_parameters(self):
        return self.__parameters

    def __batch_update(self, X_batch, y_batch, batch_size, correction_constant):
        y_pred = self.predict(X_batch)
        diff = y_batch - y_pred
        partial_reg = np.vectorize(lambda w: 0 if w == 0 else np.abs(w)/w)(self.__ws)
        partial_w = -(diff / batch_size) @ X_batch + self.__parameters.lambda_reg * partial_reg
        partial_b = -(1/batch_size) * np.sum(diff)
        self.__ws -= self.__parameters.alpha * partial_w * correction_constant
        self.__b -= self.__parameters.alpha * partial_b * correction_constant

    def __sgd_update(self, X_train, y_train):
        total_rows = len(y_train)
        batch_rows = 0
        while(batch_rows != total_rows):
            initial_index = batch_rows
            if(total_rows - batch_rows > self.__parameters.batch_size):
                final_index = batch_rows + self.__parameters.batch_size
                batch_rows += self.__parameters.batch_size
            else:
                final_index = total_rows
                batch_rows = total_rows
            X_batch = X_train.iloc[initial_index : final_index]
            y_batch = y_train.iloc[initial_index : final_index]
            correction_constant = self.__parameters.batch_size/(final_index - initial_index)
            self.__batch_update(X_batch, y_batch, self.__parameters.batch_size, correction_constant)
    
    @staticmethod
    def sigmoid(z):
        '''Takes in a float or a numpy array and returns the sigmoid of the input.'''
        return 1/(1 + np.exp(-z))