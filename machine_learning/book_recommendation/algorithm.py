import random

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def recommendation_algorithm(X, df_main, df_books, df_users_already_read, n_users):
    unique_user_ids = df_users_already_read["User-ID"].unique()
    recommendations = pd.DataFrame(columns=["User-ID", "ISBN", "Book-Title"])
    if n_users == -1:
        unique_user_ids_slice = unique_user_ids.copy()
    else:
        unique_user_ids_slice = unique_user_ids[0:n_users].copy()
    print("Computing recommendations...")
    for user_id in tqdm(unique_user_ids_slice):
        already_read = df_users_already_read[df_users_already_read["User-ID"] == user_id][
            "Already-Read-ISBN"].reset_index(drop=True)
        recommend_count = 0
        already_recommended = list(already_read)
        while recommend_count < 10:
            k = len(already_recommended) + 1
            recommendation_df = k_nearest_within_cluster(already_read.reset_index(drop=True)[0],
                                                         k, df_books)
            recommendation = recommendation_df.iloc[0]
            recommendation_list_count = 0
            while recommendation["ISBN"] in already_recommended:
                recommendation_list_count += 1
                recommendation = recommendation_df.iloc[recommendation_list_count]
            recommendation["User-ID"] = user_id
            recommendations = pd.concat([recommendations, recommendation.to_frame().T], ignore_index=True)
            already_recommended.append(recommendation["ISBN"])
            recommend_count += 1
    return recommendations


def pca_kmeans(df_main, X):
    X_reduced_pca = pca_algorithm(X)
    return kmeans_algorithm(df_main, X_reduced_pca)


def pca_algorithm(X):
    print("Computing PCA...")
    pca = PCA(n_components=0.95)
    return pd.DataFrame(pca.fit_transform(X))


def kmeans_algorithm(df_main, X_reduced_pca):
    print("Computing K-Means")
    kmeans = KMeans(n_clusters=15, n_init=1, random_state=0).fit(X_reduced_pca)
    df_books = X_reduced_pca.copy()
    df_books["Cluster"] = kmeans.labels_
    df_books["ISBN"] = df_main["ISBN"].reset_index(drop=True).copy()
    df_books["Book-Title"] = df_main["Book-Title"].reset_index(drop=True).copy()
    return df_books


def k_nearest_within_cluster(isbn, k, books_data):
    cluster = books_data[books_data["ISBN"] == isbn]["Cluster"].reset_index(drop=True)[0]
    X_train = books_data[books_data["Cluster"] == cluster]
    already_seen_clusters = [cluster]
    while len(X_train) < k:
        cluster = select_random_cluster(books_data, already_seen_clusters)
        already_seen_clusters.append(cluster)
        X_train = books_data[books_data["Cluster"] == cluster]
    neighbors = NearestNeighbors(n_neighbors=k).fit(X_train.drop(["ISBN", "Book-Title", "Cluster"], axis=1))
    distances, indices = neighbors.kneighbors(
        books_data[books_data["ISBN"] == isbn].drop(["ISBN", "Book-Title", "Cluster"], axis=1))
    flat_indices = [index for sublist in indices for index in sublist]
    return X_train.iloc[flat_indices][["ISBN", "Book-Title"]].reset_index(drop=True)


def select_random_cluster(books_data, already_seen_clusters):
    all_clusters = list(books_data["Cluster"].unique())
    while True:
        random_cluster = random.choice(all_clusters)
        if random_cluster not in already_seen_clusters:
            return random_cluster
