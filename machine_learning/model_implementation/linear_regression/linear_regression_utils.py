import os
import pandas as pd

from machine_learning.model_implementation.utils.model_implementation_utils import split_data, standard_scale, shuffle_data


def import_data():
    data_path = os.path.join('data', 'adm_data.csv')
    data = pd.read_csv(data_path)
    data.rename(columns={'Chance of Admit ': 'Chance of Admit'}, inplace=True)
    data.rename(columns={'LOR ': 'LOR'}, inplace=True)
    data.drop(["Serial No."], axis=1, inplace=True)
    return data


def preprocess(data):
    random_state = 0
    data = shuffle_data(data, random_state)
    X = data.drop(columns=["Chance of Admit"])
    y = data["Chance of Admit"]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.15, 0.15)
    X_val = standard_scale(X_val, X_train)
    X_test = standard_scale(X_test, X_train)
    X_train = standard_scale(X_train, X_train)
    return X_train, X_val, X_test, y_train, y_val, y_test
