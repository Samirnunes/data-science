import os
from preprocessing import *
from algorithm import pca_kmeans


def import_final_data():
    X_path = './data/X.csv'
    df_main_path = './data/df_main.csv'
    df_books_path = './data/df_books.csv'
    df_users_already_read_path = './data/df_users_already_read.csv'
    if (os.path.isfile(X_path) and
            os.path.isfile(df_main_path) and
            os.path.isfile(df_books_path) and
            os.path.isfile(df_users_already_read_path)):
        print("Importing saved data...")
        X = pd.read_csv(X_path)
        df_main = pd.read_csv(df_main_path)
        df_books = pd.read_csv(df_books_path)
        df_users_already_read = pd.read_csv(df_users_already_read_path)
    else:
        df_main, df_users, X = import_clean_preprocessed_data()
        df_books = pca_kmeans(df_main, X)
        df_users_already_read = preprocess_users(df_users, df_books)
        X = preprocess_X(X, df_books)
        print("Saving data...")
        X.to_csv(X_path, index=False)
        df_main.to_csv(df_main_path, index=False)
        df_books.to_csv(df_books_path, index=False)
        df_users_already_read.to_csv(df_users_already_read_path, index=False)
    return X, df_main, df_books, df_users_already_read


def import_clean_preprocessed_data():
    df_main, df_users = clean_data(*import_data())
    df_main = preprocess_main(df_main)
    X = scale_main(df_main)
    return df_main, df_users, X


def import_data():
    books = pd.read_csv("data/books.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip').drop(
        ["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1)
    ratings = pd.read_csv("data/ratings.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip')
    users = pd.read_csv("data/users.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip')
    return books, ratings, users
