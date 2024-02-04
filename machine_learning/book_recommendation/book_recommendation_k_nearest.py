import random
from sklearn.neighbors import NearestNeighbors


class BookRecommendationKNearest:
    def __init__(self):
        pass

    @staticmethod
    def k_nearest_within_cluster(isbn, k, books_data):
        cluster = books_data[books_data["ISBN"] == isbn]["Cluster"].reset_index(drop=True)[0]
        X_train = books_data[books_data["Cluster"] == cluster]
        already_seen_clusters = [cluster]
        while len(X_train) < k:
            cluster = BookRecommendationKNearest.select_random_cluster(books_data, already_seen_clusters)
            already_seen_clusters.append(cluster)
            X_train = books_data[books_data["Cluster"] == cluster]
        neighbors = NearestNeighbors(n_neighbors=k).fit(X_train.drop(["ISBN", "Book-Title", "Cluster"], axis=1))
        distances, indices = neighbors.kneighbors(
            books_data[books_data["ISBN"] == isbn].drop(["ISBN", "Book-Title", "Cluster"], axis=1))
        flat_indices = [index for sublist in indices for index in sublist]
        return X_train.iloc[flat_indices][["ISBN", "Book-Title"]].reset_index(drop=True)

    @staticmethod
    def select_random_cluster(books_data, already_seen_clusters):
        all_clusters = list(books_data["Cluster"].unique())
        while True:
            random_cluster = random.choice(all_clusters)
            if random_cluster not in already_seen_clusters:
                return random_cluster
