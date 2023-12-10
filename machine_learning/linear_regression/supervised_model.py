from abc import ABC, abstractmethod

class SupervisedModel(ABC):
    @abstractmethod
    def __init__(self, parameters: dict):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def loss(self, X, y):
        pass

    @abstractmethod
    def predict(self, X_pred):
        pass

    @abstractmethod
    def get_train_loss(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

