import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def shuffle_data(data, random_state):
    '''Shuffles a Pandas Dataframe's data.'''
    rand = np.random.RandomState(random_state)
    return data.reindex(rand.permutation(data.index))


def standard_scale(X, X_train):
    mean = X_train.mean()
    std = X_train.std()
    return (X - mean) / std


def min_max_scale(feature, unscaled_train_feature):
    '''Scales a feature so that its values lie between 0 and 1.'''
    minimum = min(unscaled_train_feature)
    maximum = max(unscaled_train_feature)
    return (feature - minimum) / (maximum - minimum)


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


def quadratic_loss(model, X_test, y_test):
    return np.mean((model.predict(X_test) - y_test) ** 2)


def log_loss(self, X_test, y_test):
    predictions = self.predict(X_test)
    fst_term = y_test * np.log(predictions)
    sec_term = (1 - y_test) * np.log(1 - predictions)
    return -np.mean(fst_term + sec_term)


def plot_correlations(X_train, y_train):
    sns.heatmap(pd.concat([X_train, y_train], axis=1).corr(), cmap="flare", annot=True)
    plt.show()


def plot_train_loss(loss):
    plt.plot(loss)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
