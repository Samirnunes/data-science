from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class SupervisedModel(ABC):
    @abstractmethod
    def __init__(self, parameters: dict):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_pred):
        pass