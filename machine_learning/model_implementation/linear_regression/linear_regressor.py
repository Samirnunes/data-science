import numpy as np
from copy import deepcopy
from machine_learning.model_implementation.linear_regression.linear_regression_parameters import \
    LinearRegressionParameters
from machine_learning.model_implementation.base_classes.supervised_model import SupervisedModel


class LinearRegressor(SupervisedModel):
    def __init__(self, parameters: LinearRegressionParameters):
        self._parameters = deepcopy(parameters)
        self._ws = np.array(deepcopy(self._parameters.initial_weights))
        self._b = deepcopy(self._parameters.initial_bias)
        self._train_loss = []

    def fit(self, X_train, y_train, print_loss=False):
        for _ in range(0, self._parameters.epochs):
            self._sgd_update(X_train, y_train)
            loss = self.loss(X_train, y_train)
            self._train_loss.append(loss)
            if print_loss:
                print(f'loss = {loss}')

    def loss(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def predict(self, X_pred):
        return X_pred @ self._ws + self._b

    def get_weights(self):
        return self._ws

    def get_bias(self):
        return self._b

    def get_train_loss(self):
        return self._train_loss

    def get_parameters(self):
        return self._parameters

    def _sgd_update(self, X_train, y_train):
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
            X_batch = X_train.iloc[initial_index:final_index].copy()
            y_batch = y_train.iloc[initial_index:final_index].copy()
            correction_constant = self._parameters.batch_size / (final_index - initial_index)
            self._batch_update(X_batch, y_batch, self._parameters.batch_size, correction_constant)

    def _batch_update(self, X_batch, y_batch, batch_size, correction_constant):
        y_pred = self.predict(X_batch)
        diff = y_batch - y_pred
        partial_w = -(2 / batch_size) * (diff @ X_batch.values) + self._partial_l2() + self._partial_l1()
        partial_b = -(2 / batch_size) * np.sum(diff)
        self._ws -= self._parameters.alpha * partial_w * correction_constant
        self._b -= self._parameters.alpha * partial_b * correction_constant

    def _partial_l2(self):
        return 2 * self._parameters.lambda_reg * self._ws

    def _partial_l1(self):
        def sign(xs):
            sign_lambda = lambda x: 1 if x > 0 else -1 if x < 0 else 0
            return np.array(list(map(sign_lambda, xs)))
        return self._parameters.gama_reg * sign(self._ws)
