import numpy as np
from copy import deepcopy
from logistic_regression_parameters import LogisticRegressionParameters
from machine_learning.model_implementation.base_classes.supervised_model import SupervisedModel


class LogisticRegressor(SupervisedModel):
    def __init__(self, parameters: LogisticRegressionParameters):
        self._parameters = deepcopy(parameters)
        self._ws = deepcopy(self._parameters.initial_weights)
        self._b = deepcopy(self._parameters.initial_bias)
        self._train_loss = []

    def fit(self, X_train, y_train, print_loss=False):
        for _ in range(0, self._parameters.epochs):
            self.__sgd_update(X_train, y_train)
            loss = self.loss(X_train, y_train)
            self._train_loss.append(loss)
            if print_loss:
                print(f'loss = {loss}')

    def loss(self, X, y):
        predictions = self.predict(X)
        fst_term = y * np.log(predictions)
        sec_term = (1 - y) * np.log(1 - predictions)
        return -np.mean(fst_term + sec_term)

    def predict(self, X_pred):
        linear_predictions = X_pred.mul(self._ws).sum(axis=1) + self._b
        return np.array(LogisticRegressor.sigmoid(linear_predictions))

    def get_weights(self):
        return self._ws

    def get_bias(self):
        return self._b

    def get_train_loss(self):
        return self._train_loss

    def get_parameters(self):
        return self._parameters

    def __batch_update(self, X_batch, y_batch, batch_size, correction_constant):
        y_pred = self.predict(X_batch)
        diff = y_batch - y_pred
        partial_w = -(diff / batch_size) @ X_batch + self._partial_l2() + self._partial_l1()
        partial_b = -(1 / batch_size) * np.sum(diff)
        self._ws -= self._parameters.alpha * partial_w * correction_constant
        self._b -= self._parameters.alpha * partial_b * correction_constant

    def __sgd_update(self, X_train, y_train):
        total_rows = len(y_train)
        batch_rows = 0
        while batch_rows != total_rows:
            initial_index = batch_rows
            if total_rows - batch_rows > self._parameters.batch_size:
                final_index = batch_rows + self._parameters.batch_size
                batch_rows += self._parameters.batch_size
            else:
                final_index = total_rows
                batch_rows = total_rows
            X_batch = X_train.iloc[initial_index: final_index]
            y_batch = y_train.iloc[initial_index: final_index]
            correction_constant = self._parameters.batch_size / (final_index - initial_index)
            self.__batch_update(X_batch, y_batch, self._parameters.batch_size, correction_constant)

    @staticmethod
    def sigmoid(z):
        """Takes in a float or a numpy array and returns the sigmoid of the input."""
        return 1 / (1 + np.exp(-z))

    def _partial_l2(self):
        return 2 * self._parameters.lambda_reg * self._ws

    def _partial_l1(self):
        def sign(xs):
            sign_lambda = lambda x: 1 if x > 0 else -1 if x < 0 else 0
            return np.array(list(map(sign_lambda, xs)))

        return self._parameters.gamma_reg * sign(self._ws)
