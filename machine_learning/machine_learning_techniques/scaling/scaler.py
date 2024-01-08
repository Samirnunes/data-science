import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)


class Scaler:
    def __init__(self, scale_method: str):
        self.scale_method = scale_method

    def scale(self, X_train, X_test, num_cols):
        X_train_num = X_train[num_cols].copy()
        X_test_num = X_test[num_cols].copy()
        scaler = self.__select_scaler()
        X_train[num_cols] = pd.DataFrame(scaler.fit_transform(X_train_num), columns=X_train_num.columns)
        X_test[num_cols] = pd.DataFrame(scaler.transform(X_test_num), columns=X_train_num.columns)
        return X_train, X_test

    def __select_scaler(self):
        if self.scale_method.lower() == "minmax":
            return MinMaxScaler()
        if self.scale_method.lower() == "robust":
            return RobustScaler()
        if self.scale_method.lower() == "standard":
            return StandardScaler()
        return StandardScaler()
