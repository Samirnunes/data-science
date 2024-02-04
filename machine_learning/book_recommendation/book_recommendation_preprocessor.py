import pandas as pd
from sklearn.preprocessing import StandardScaler
from book_main_step_preprocessor import BookMainStepPreprocessor
from book_recommendation_pca import BookRecommendationPCA
from book_recommendation_k_means import BookRecommendationKMeans

class BookRecommendationPreprocessor:
    def __init__(self):
        pass

    def fit_transform(self, books, users, ratings):
        self.fit(books, users, ratings)
        return self.transform(books, users, ratings)

    def fit(self, books, users, ratings):
        return self

    def transform(self, books, users, ratings):
        df_main, df_users = BookRecommendationPreprocessor.clean_step(books, users, ratings)
        df_main = BookMainStepPreprocessor().fit_transform(df_main)
        X_train = BookRecommendationPreprocessor.generate_X_train_for_algorithms(df_main)
        X_train_reduced_pca = BookRecommendationPCA.pca(X_train)
        df_books = BookRecommendationKMeans.kmeans(df_main, X_train_reduced_pca)
        df_users_already_read = BookRecommendationPreprocessor.generate_users_already_read(df_users, df_books)
        X_train = BookRecommendationPreprocessor.add_clusters_to_X_train(X_train, df_books)
        return df_main, df_books, df_users_already_read, X_train

    @staticmethod
    def clean_step(books, users, ratings):
        books = books.dropna()
        users["Age"] = users["Age"].fillna(users["Age"].median())
        users_ratings = pd.merge(users, ratings, on="User-ID", how="inner")
        books_users_ratings = pd.merge(books, users_ratings, on="ISBN", how="inner")
        df_main = books_users_ratings.copy()
        df_users = books_users_ratings.copy()
        return df_main, df_users

    @staticmethod
    def generate_X_train_for_algorithms(df_main):
        print("Generating X_train for algorithms...")
        X_train = df_main.drop(["ISBN", 'Book-Title'], axis=1)
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    @staticmethod
    def generate_users_already_read(df_users, df_books):
        print("Preprocessing users already read books data...")
        df_users = pd.merge(df_users, df_books[["ISBN", "Cluster"]], on="ISBN", how="inner")
        df_users = df_users[["User-ID", "ISBN", "Cluster"]]
        df_users_already_read = df_users.groupby("User-ID")["ISBN"].value_counts()
        return pd.DataFrame(df_users_already_read.index.to_list(),
                            columns=["User-ID", "Already-Read-ISBN"])

    @staticmethod
    def add_clusters_to_X_train(X_train, df_books):
        print("Adding clusters to X_train data...")
        X_train["Cluster"] = df_books["Cluster"].copy()
        return X_train
