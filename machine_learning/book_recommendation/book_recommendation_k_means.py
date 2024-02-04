from sklearn.cluster import KMeans
class BookRecommendationKMeans():
    def __init__(self):
        pass

    @staticmethod
    def kmeans(df_main, X_reduced_pca):
        print("Computing K-Means...")
        kmeans = KMeans(n_clusters=11, n_init=1, random_state=0).fit(X_reduced_pca)
        df_books = X_reduced_pca.copy()
        df_books["Cluster"] = kmeans.labels_
        df_books["ISBN"] = df_main["ISBN"].reset_index(drop=True).copy()
        df_books["Book-Title"] = df_main["Book-Title"].reset_index(drop=True).copy()
        return df_books
