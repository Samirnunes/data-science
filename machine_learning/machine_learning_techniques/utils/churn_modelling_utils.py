import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from machine_learning.machine_learning_techniques.scaling.scaler import Scaler


def import_data():
    df = pd.read_csv("../data/Churn_Modelling.csv")
    return df.drop(columns=["RowNumber", "CustomerId", "Surname"])


def split_data(df):
    X = df.loc[:, ~df.columns.isin(["Exited"])]
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)
    return X_train, X_test, y_train, y_test


def impute_data(X_train, X_test, y_train, y_test):
    imputer = SimpleImputer(strategy="most_frequent")
    cols = X_train.columns
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=cols)
    return X_train, X_test, y_train, y_test


def num_cat_cols(X):
    cat_cols = ["Geography", "Gender"]
    num_cols = list(set(X.columns).difference(cat_cols))
    return num_cols, cat_cols


def one_hot_encode(X_train, X_test, y_train, y_test):
    _, cat_cols = num_cat_cols(X_train)
    one_hot_train = pd.get_dummies(X_train[cat_cols])
    one_hot_test = pd.get_dummies(X_test[cat_cols])
    X_train.drop(cat_cols, axis=1, inplace=True)
    X_test.drop(cat_cols, axis=1, inplace=True)
    X_train = pd.concat([X_train, one_hot_train], axis=1)
    X_test = pd.concat([X_test, one_hot_test], axis=1)

    return X_train, X_test, y_train, y_test


def preprocess_without_scale(df):
    return one_hot_encode(*impute_data(*split_data(df)))


def preprocess(df):
    X_train, X_test, y_train, y_test = preprocess_without_scale(df)
    num_cols, _ = num_cat_cols(X_train)
    scaler = Scaler("standard")
    X_train, X_test = scaler.scale(X_train, X_test, num_cols)
    return X_train, X_test, y_train, y_test
