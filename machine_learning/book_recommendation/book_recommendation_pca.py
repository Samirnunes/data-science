import pandas as pd
from sklearn.decomposition import PCA
class BookRecommendationPCA():
    def __init__(self):
        pass

    @staticmethod
    def pca(X):
        print("Computing PCA...")
        pca = PCA(n_components=0.95)
        return pd.DataFrame(pca.fit_transform(X))