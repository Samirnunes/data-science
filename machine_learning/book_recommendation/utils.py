import os
import pandas as pd
from book_recommendation_preprocessor import BookRecommendationPreprocessor

def import_final_data():
    X_path = './data/X_train.csv'
    df_main_path = './data/df_main.csv'
    df_books_path = './data/df_books.csv'
    df_users_already_read_path = './data/df_users_already_read.csv'
    if (os.path.isfile(X_path) and
            os.path.isfile(df_main_path) and
            os.path.isfile(df_books_path) and
            os.path.isfile(df_users_already_read_path)):
        print("Importing saved data...")
        X_train = pd.read_csv(X_path)
        df_main = pd.read_csv(df_main_path)
        df_books = pd.read_csv(df_books_path)
        df_users_already_read = pd.read_csv(df_users_already_read_path)
    else:
        books, ratings, users = import_books_ratings_users()
        df_main, df_books, df_users_already_read, X_train = (BookRecommendationPreprocessor()
                                                             .fit_transform(books, users, ratings))
        print("Saving data...")
        X_train.to_csv(X_path, index=False)
        df_main.to_csv(df_main_path, index=False)
        df_books.to_csv(df_books_path, index=False)
        df_users_already_read.to_csv(df_users_already_read_path, index=False)
    return df_books, df_users_already_read, df_main, X_train


def import_books_ratings_users():
    books = pd.read_csv("data/books.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip').drop(
        ["Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1)
    ratings = pd.read_csv("data/ratings.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip')
    users = pd.read_csv("data/users.csv", sep=";", encoding="latin-1", low_memory=False, on_bad_lines='skip')
    return books, ratings, users
