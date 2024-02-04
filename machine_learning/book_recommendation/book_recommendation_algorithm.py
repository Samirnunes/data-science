import os
import pandas as pd
from tqdm import tqdm
from book_recommendation_k_nearest import BookRecommendationKNearest


class BookRecommendationAlgorithm:
    def __init__(self):
        self.recommendations_path = "./data/recommendations.csv"
        self.df_books = None
        self.df_users_already_read = None

    def fit_recommend(self, df_books, df_users_already_read, n_users):
        if os.path.isfile(self.recommendations_path):
            recommendations = pd.read_csv(self.recommendations_path)
        else:
            self.fit(df_books, df_users_already_read)
            recommendations = self.recommend(n_users)
            recommendations.to_csv(self.recommendations_path, index=False)
        return recommendations

    def fit(self, df_books, df_users_already_read):
        self.df_books = df_books
        self.df_users_already_read = df_users_already_read
        return self

    def recommend(self, n_users):
        unique_user_ids = self.df_users_already_read["User-ID"].unique()
        recommendations = pd.DataFrame(columns=["User-ID", "ISBN", "Book-Title"])
        if n_users == -1:
            unique_user_ids_slice = unique_user_ids.copy()
        else:
            unique_user_ids_slice = unique_user_ids[0:n_users].copy()
        print("Computing recommendations...")
        for user_id in tqdm(unique_user_ids_slice):
            already_read = self.df_users_already_read[self.df_users_already_read["User-ID"] == user_id][
                "Already-Read-ISBN"].reset_index(drop=True)
            recommend_count = 0
            already_recommended = list(already_read)
            while recommend_count < 10:
                k = len(already_recommended) + 1
                recommendation_df = BookRecommendationKNearest.k_nearest_within_cluster(
                    already_read.reset_index(drop=True)[0], k, self.df_books
                )
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

