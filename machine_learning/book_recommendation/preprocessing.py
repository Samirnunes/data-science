import pandas as pd
from category_encoders import CountEncoder
from sklearn.preprocessing import StandardScaler


def clean_data(books, ratings, users):
    print("Cleaning data...")
    books = books.dropna()
    users["Age"] = users["Age"].fillna(users["Age"].median())
    users_ratings = pd.merge(users, ratings, on="User-ID", how="inner")
    books_users_ratings = pd.merge(books, users_ratings, on="ISBN", how="inner")
    df_main = books_users_ratings.copy()
    df_users = books_users_ratings.copy()
    return df_main, df_users


def preprocess_main(df_main):
    print("Preprocessing main data...")
    df_main = df_main[df_main["Age"] < 100]
    df_main = df_main[df_main["Year-Of-Publication"].str.isnumeric()]
    df_main["Year-Of-Publication"] = df_main["Year-Of-Publication"].apply(lambda x: int(x))
    df_main = df_main[df_main["Year-Of-Publication"] <= 2004]

    mean_age_by_book = df_main.groupby("Book-Title")["Age"].mean().rename("Mean-Age-By-Book")
    df_main = pd.merge(df_main, mean_age_by_book, on="Book-Title", how="inner")
    mean_age_by_author = df_main.groupby("Book-Author")["Age"].mean().rename("Mean-Age-By-Author")
    df_main = pd.merge(df_main, mean_age_by_author, on="Book-Author", how="inner")
    mean_age_by_publisher = df_main.groupby("Publisher")["Age"].mean().rename("Mean-Age-By-Publisher")
    df_main = pd.merge(df_main, mean_age_by_publisher, on="Publisher", how="inner")
    mean_rating_by_book = df_main.groupby("Book-Title")["Book-Rating"].mean().rename("Mean-Rating-By-Book")
    df_main = pd.merge(df_main, mean_rating_by_book, on="Book-Title", how="inner")
    mean_rating_by_author = df_main.groupby("Book-Author")["Book-Rating"].mean().rename("Mean-Rating-By-Author")
    df_main = pd.merge(df_main, mean_rating_by_author, on="Book-Author", how="inner")
    mean_rating_by_publisher = df_main.groupby("Publisher")["Book-Rating"].mean().rename("Mean-Rating-By-Publisher")
    df_main = pd.merge(df_main, mean_rating_by_publisher, on="Publisher", how="inner")
    df_main = df_main.drop(["Age", "Book-Rating", "User-ID"], axis=1)

    location_count_by_book = df_main.groupby(["ISBN", "Location"]).size().reset_index(
        name="Readers-Count-In-Top-Location")
    top_location_by_book = location_count_by_book.loc[
        location_count_by_book.groupby("ISBN")["Readers-Count-In-Top-Location"].idxmax()].rename(
        {"Location": "Top-Location-By-Book"}, axis=1)
    df_main = pd.merge(df_main, top_location_by_book, on="ISBN", how="inner")

    location_encoder = CountEncoder()
    df_main["Location-Encoded"] = location_encoder.fit_transform(df_main["Location"])
    location_encodings = df_main.groupby(["Location"])["Location-Encoded"].max().rename("Top-Location-By-Book-Encoded")
    df_main = df_main.drop(["Location-Encoded"], axis=1)
    df_main = pd.merge(df_main, location_encodings, left_on="Top-Location-By-Book", right_on="Location", how="inner")
    df_main = df_main.drop(["Location", "Top-Location-By-Book"], axis=1)
    author_encoder = CountEncoder()
    df_main["Book-Author-Encoded"] = author_encoder.fit_transform(df_main["Book-Author"])
    df_main = df_main.drop(["Book-Author"], axis=1)
    publisher_encoder = CountEncoder()
    df_main["Publisher-Encoded"] = publisher_encoder.fit_transform(df_main["Publisher"])
    df_main = df_main.drop(["Publisher"], axis=1)

    df_main["Book-Contribution-To-Top-Location"] = df_main["Readers-Count-In-Top-Location"] / df_main[
        "Top-Location-By-Book-Encoded"]
    df_main = df_main.drop(["Readers-Count-In-Top-Location"], axis=1)

    isbn_encoder = CountEncoder()
    df_main["Book-Appearances"] = isbn_encoder.fit_transform(df_main["ISBN"])

    df_main = df_main.drop_duplicates()

    return df_main


def scale_main(df_main):
    print("Scaling main data...")
    X = df_main.drop(["ISBN", 'Book-Title'], axis=1)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


def preprocess_users(df_users, df_books):
    print("Preprocessing users data...")
    df_users = pd.merge(df_users, df_books[["ISBN", "Cluster"]], on="ISBN", how="inner")
    df_users = df_users[["User-ID", "ISBN", "Cluster"]]
    df_users_already_read = df_users.groupby("User-ID")["ISBN"].value_counts()
    return pd.DataFrame(df_users_already_read.index.to_list(),
                        columns=["User-ID", "Already-Read-ISBN"])


def preprocess_X(X, df_books):
    print("Preprocessing X data...")
    X["Cluster"] = df_books["Cluster"].copy()
    return X
