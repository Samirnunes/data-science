import numpy as np
from supervised_model import *
from parameters import Parameters
from copy import deepcopy

class LinearRegressor(SupervisedModel):
    def __init__(self, parameters: Parameters):
        self.__parameters = deepcopy(parameters)
        self.__parameters.ws = deepcopy(self.__parameters.initial_weights)
        self.__parameters.b = deepcopy(self.__parameters.initial_bias)
        self.__train_loss = []

    def fit(self, X_train, y_train, print_loss = False):
        for _ in range(0, self.__parameters.epochs):
            self.__sgd_update(X_train, y_train)
            loss = self.loss(X_train, y_train)
            self.__train_loss.append(loss)
            if print_loss:
                print(f'loss = {loss}')

    def loss(self, X, y):
        return np.mean((self.predict(X) - y)**2)
    
    def predict(self, X_pred):
        return np.array(X_pred @ self.__parameters.ws + self.__parameters.b)
    
    def get_train_loss(self):
        return self.__train_loss
    
    def get_parameters(self):
        return self.__parameters

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

    def __batch_update(self, X_batch, y_batch, batch_size, correction_constant):
        y_pred = self.predict(X_batch)
        diff = y_pred - y_batch
        partial_w = (2/batch_size) * (diff @ X_batch.values) + 2 * self.__parameters.lambda_reg * self.__parameters.ws
        partial_b = (2/batch_size) * np.sum(diff)
        self.__parameters.ws -= self.__parameters.alpha * partial_w * correction_constant
        self.__parameters.b -= self.__parameters.alpha * partial_b * correction_constant




