from category_encoders import CountEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class BookMainStepPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.df_main = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return BookMainStepPreprocessor.main_step(X)

    @staticmethod
    def main_step(df_main):
        print("Preprocessing main data...")
        df_main = BookMainStepPreprocessor.preprocess_year_of_publication(df_main)
        df_main = BookMainStepPreprocessor.preprocess_age(df_main)
        df_main = BookMainStepPreprocessor.generate_age_features(df_main)
        df_main = BookMainStepPreprocessor.generate_rating_features(df_main)
        df_main = BookMainStepPreprocessor.generate_top_locations(df_main)
        df_main = BookMainStepPreprocessor.generate_top_locations_encoding(df_main)
        df_main = BookMainStepPreprocessor.generate_author_encoding(df_main)
        df_main = BookMainStepPreprocessor.generate_publisher_encoding(df_main)
        df_main = BookMainStepPreprocessor.generate_book_contribution_to_top_location(df_main)
        df_main = BookMainStepPreprocessor.generate_book_appearances(df_main)
        return df_main.drop_duplicates()

    @staticmethod
    def preprocess_year_of_publication(df_main):
        df_main = df_main[df_main["Year-Of-Publication"].str.isnumeric()]
        df_main.loc[:, "Year-Of-Publication"] = df_main["Year-Of-Publication"].apply(lambda x: int(x))
        return df_main[df_main["Year-Of-Publication"] <= 2004]

    @staticmethod
    def preprocess_age(df_main):
        return df_main[(df_main["Age"] >= 18) & (df_main["Age"] < 100)]

    @staticmethod
    def generate_age_features(df_main):
        mean_age_by_book = df_main.groupby("Book-Title")["Age"].mean().rename("Mean-Age-By-Book")
        df_main = pd.merge(df_main, mean_age_by_book, on="Book-Title", how="inner")
        mean_age_by_author = df_main.groupby("Book-Author")["Age"].mean().rename("Mean-Age-By-Author")
        df_main = pd.merge(df_main, mean_age_by_author, on="Book-Author", how="inner")
        mean_age_by_publisher = df_main.groupby("Publisher")["Age"].mean().rename("Mean-Age-By-Publisher")
        df_main = pd.merge(df_main, mean_age_by_publisher, on="Publisher", how="inner")
        return df_main

    @staticmethod
    def generate_rating_features(df_main):
        mean_rating_by_book = df_main.groupby("Book-Title")["Book-Rating"].mean().rename("Mean-Rating-By-Book")
        df_main = pd.merge(df_main, mean_rating_by_book, on="Book-Title", how="inner")
        mean_rating_by_author = df_main.groupby("Book-Author")["Book-Rating"].mean().rename("Mean-Rating-By-Author")
        df_main = pd.merge(df_main, mean_rating_by_author, on="Book-Author", how="inner")
        mean_rating_by_publisher = df_main.groupby("Publisher")["Book-Rating"].mean().rename("Mean-Rating-By-Publisher")
        df_main = pd.merge(df_main, mean_rating_by_publisher, on="Publisher", how="inner")
        return df_main.drop(["Age", "Book-Rating", "User-ID"], axis=1)

    @staticmethod
    def generate_top_locations(df_main):
        location_count_by_book = df_main.groupby(["ISBN", "Location"]).size().reset_index(
            name="Readers-Count-In-Top-Location")
        top_location_by_book = location_count_by_book.loc[
            location_count_by_book.groupby("ISBN")["Readers-Count-In-Top-Location"].idxmax()].rename(
            {"Location": "Top-Location-By-Book"}, axis=1)
        return pd.merge(df_main, top_location_by_book, on="ISBN", how="inner")

    @staticmethod
    def generate_top_locations_encoding(df_main):
        location_encoder = CountEncoder()
        df_main["Location-Encoded"] = location_encoder.fit_transform(df_main["Location"])
        location_encodings = df_main.groupby(["Location"])["Location-Encoded"].max().rename(
            "Top-Location-By-Book-Encoded")
        df_main = df_main.drop(["Location-Encoded"], axis=1)
        df_main = pd.merge(df_main, location_encodings, left_on="Top-Location-By-Book", right_on="Location",
                           how="inner")
        return df_main.drop(["Location", "Top-Location-By-Book"], axis=1)

    @staticmethod
    def generate_author_encoding(df_main):
        author_encoder = CountEncoder()
        df_main["Book-Author-Encoded"] = author_encoder.fit_transform(df_main["Book-Author"])
        return df_main.drop(["Book-Author"], axis=1)

    @staticmethod
    def generate_publisher_encoding(df_main):
        publisher_encoder = CountEncoder()
        df_main["Publisher-Encoded"] = publisher_encoder.fit_transform(df_main["Publisher"])
        return df_main.drop(["Publisher"], axis=1)

    @staticmethod
    def generate_book_contribution_to_top_location(df_main):
        df_main["Book-Contribution-To-Top-Location"] = df_main["Readers-Count-In-Top-Location"] / df_main[
            "Top-Location-By-Book-Encoded"]
        return df_main.drop(["Readers-Count-In-Top-Location"], axis=1)

    @staticmethod
    def generate_book_appearances(df_main):
        isbn_encoder = CountEncoder()
        df_main["Book-Appearances"] = isbn_encoder.fit_transform(df_main["ISBN"])
        return df_main
