import pandas as pd


def import_data():
    train_data = pd.read_csv('data/train.csv')
    X_test = pd.read_csv('data/test.csv')

    X_train = train_data.loc[:, train_data.columns != 'Survived']
    y_train = train_data['Survived']

    return X_train, X_test, y_train
