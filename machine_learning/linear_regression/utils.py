import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data, random_state):
    '''Shuffles a Pandas Dataframe's data.'''
    rand = np.random.RandomState(random_state)
    return data.reindex(rand.permutation(data.index))

def standard_scale(X, X_train):
    mean = X_train.mean()
    std = X_train.std()
    return (X - mean)/std

def split_data(X, y, test_split_factor: float, val_split_factor: float):
    total_rows = len(y)
    test_size = int(test_split_factor * total_rows)
    val_size = int(val_split_factor * total_rows)
    X_test = X.iloc[0:test_size]
    y_test = y.iloc[0:test_size]
    X_val = X.iloc[test_size:test_size + val_size]
    y_val = y.iloc[test_size:test_size + val_size]
    X_train = X.iloc[test_size + val_size:total_rows]
    y_train = y.iloc[test_size + val_size:total_rows]
    return X_train, X_val, X_test, y_train, y_val, y_test

def loss(model, X_test, y_test):
    return np.mean((model.predict(X_test) - y_test)**2)

def plot_train_loss(loss):
    plt.plot(loss)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")