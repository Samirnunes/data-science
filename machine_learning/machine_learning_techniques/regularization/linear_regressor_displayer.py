from copy import deepcopy
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

from machine_learning.model_implementation.linear_regression.linear_regression_parameters import \
    LinearRegressionParameters
from machine_learning.model_implementation.linear_regression.linear_regressor import LinearRegressor


class LinearRegressorDisplayer(LinearRegressor):

    def __init__(self, parameters: LinearRegressionParameters):
        super().__init__(parameters)
        self.__ws_history = [deepcopy(self._ws)]
        self.__b_history = [deepcopy(self._b)]

    def fit_store(self, X_train, y_train, print_loss=False, only_reg=False):
        for _ in range(0, self._parameters.epochs):
            if only_reg:
                self._sgd_update_only_reg(X_train, y_train)
            else:
                self._sgd_update(X_train, y_train)
            loss = self.loss(X_train, y_train)
            self._train_loss.append(loss)
            self.__ws_history.append(deepcopy(self._ws))
            self.__b_history.append(deepcopy(self._b))
            if print_loss:
                print(f'loss = {loss}')

    def _sgd_update_only_reg(self, X_train, y_train):
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
            self._batch_update_only_reg(X_batch, y_batch, self._parameters.batch_size, correction_constant)

    def _batch_update_only_reg(self, X_batch, y_batch, batch_size, correction_constant):
        partial_w = self._partial_l2() + self._partial_l1()
        partial_b = 0
        self._ws -= self._parameters.alpha * partial_w
        self._b -= self._parameters.alpha * partial_b

    def get_ws_history(self):
        return self.__ws_history

    def get_b_history(self):
        return self.__b_history

    def display_ws_history(self):
        data = self.__ws_history
        index_slider = LinearRegressorDisplayer.index_slider(self.__ws_history)

        def update_plot(index):
            plt.figure(figsize=(12, 6))
            plt.clf()
            plt.bar(range(0, len(data[index])), data[index])
            plt.title(f"Weights during train")
            plt.xlabel("")
            plt.ylabel("Weights")
            plt.grid(True)
            plt.show()

        display(index_slider, LinearRegressorDisplayer.output_plot(update_plot, index_slider))

    @staticmethod
    def index_slider(data):
        return widgets.IntSlider(value=0, min=0, max=len(data) - 1, description='Epoch')

    @staticmethod
    def output_plot(update_plot, index_slider):
        return widgets.interactive_output(update_plot, {'index': index_slider})
