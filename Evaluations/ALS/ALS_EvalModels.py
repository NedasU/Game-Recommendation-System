# This code evaluates the performance of the ALS Model. To use this file, you only need to run the file.
# Changing the value of K can change the number of top-k recommendations are evaluated.
# Additionally, to change the selected model that needs to be evaluated, the dataset and the model need to change:
# The dataset should be this for playtime: "../../Datasets/playtime versions/User-Item-Interaction_2651.csv"
# For the playtime version, the ALS model that is commented out should be uncommented. They are the grid searched best
# parameters for the playtime version.
# The default version here evaluates the Achievement version already.

import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import collect_list, struct
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# initialises the spark session
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
start_time = time.time()

spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.driver.memory", "6g")\
    .getOrCreate()

#Loads the dataset and traintest split. ALso normalises the popularity and the genres for each game.
dataset_path = "../../Datasets/User-Item-Interaction_2651.csv"
data = pd.read_csv(dataset_path)
gameinfo_df = pd.read_csv("../../Datasets/Hybrid/Hybrid_Game_File.csv")
df = spark.read.csv(dataset_path, schema='user_id int, game_id int, rating int', header=True)

dfs_train, dfs_test = df.randomSplit([0.8, 0.2], seed=42)

item_popularity = data['game_id'].value_counts().reset_index()
item_popularity.columns = ['game_id', 'popularity']

scaler = MinMaxScaler()
item_popularity['popularity_normalized'] = scaler.fit_transform(item_popularity[['popularity']])
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# Remove NaN values from the 'genres' column
gameinfo_df.dropna(subset=['genres'], inplace=True)
diversity_genres = gameinfo_df['genres']
# Initialize and fit TF-IDF vectorizer
processed_genres = tfidf_vectorizer.fit_transform(diversity_genres)

USER_COL = "user_id"
ITEM_COL = "game_id"
RATING_COL = "rating"
PREDICTION_COL = "prediction"

# These 2 ALS models are here because I could not manage to save the models. This may have been a hardware issue.

# Best parameters 0-5-2651_Achievement:
als = ALS(
    userCol=USER_COL,
    itemCol=ITEM_COL,
    ratingCol=RATING_COL,
    coldStartStrategy='drop',
    nonnegative=True,
    implicitPrefs=False,
    seed=42,
    regParam=0.1,
    maxIter=20,
    rank=20
)

# Uncomment this to use the playtime version instead:
# Best parameters: 0-5-2651_playtime:
# als = ALS(
#     userCol=USER_COL,
#     itemCol=ITEM_COL,
#     ratingCol=RATING_COL,
#     coldStartStrategy='drop',
#     nonnegative=True,
#     implicitPrefs=False,
#     seed=42,
#     regParam=0.1,
#     maxIter=20,
#     rank=30
# )


# Fits the model to the training data
print("Fitting model to the data for cross validation and hyper parameter tuning")
model = als.fit(dfs_train)

# Forms predictions based on the test set
dfs_pred = model.transform(dfs_test)

# Groups the predicted ratings by user ids. Iterates over each user and for each user it iterates over the top-k
# games and calculates its novelty through the normalised popularity score. The mean average of the novelty scores
# through all users is returned
def novelty_at_k(predictions, k=10):
    novelty_scores = []

    grouped_predictions = predictions.groupBy("user_id").agg(
        collect_list(struct("game_id", "rating", "prediction")).alias("predictions"))

    for row in grouped_predictions.collect():
        user_ratings = row["predictions"]
        user_ratings.sort(key=lambda x: x['prediction'], reverse=True)
        ids = [ro['game_id'] for ro in user_ratings[:k]]
        user_novelty_scores = []
        for iid in ids:
            pop_score = item_popularity[item_popularity['game_id'] == iid]['popularity_normalized'].values
            novelty = 1 - pop_score[0] if len(pop_score) > 0 else 0
            user_novelty_scores.append(novelty)
        avg_novelty_user = np.mean(user_novelty_scores)
        novelty_scores.append(avg_novelty_user)

    return np.mean(novelty_scores)


# Takes the recommended games ids and gets their genres. pairwise dissimilarity is calculated between each
# recommended item. The mean value of all these diversity values are returned as the overall diversity score.
def intra_list_dissimilarity(recommended_game_ids):
    recommended_game_ids = [game_id for game_id in recommended_game_ids if game_id in gameinfo_df['id'].unique()]
    genres = processed_genres[gameinfo_df['id'].isin(recommended_game_ids)]

    dissimilarities = []
    for i in range(len(recommended_game_ids)):
        for j in range(i + 1, len(recommended_game_ids)):
            game1 = genres[i]
            game2 = genres[j]
            dissimilarity = 1 - cosine_similarity(game1.reshape(1, -1), game2.reshape(1, -1))
            dissimilarities.append(dissimilarity)

    avg_dissimilarity = np.mean(dissimilarities)
    return avg_dissimilarity


# This function is used to calculate the precision and recall at k based on a threshold.
# predicted ratings is taken in and all the top-k recommendations predicted ratings are matched against the threshold
# If the predicted rating is above the threshold it adds to the sum. Returns the precision and recall of each user.
def precision_recall_diversity_at_k(predictions, k=10, threshold=3):
    # Group predictions by user_id
    grouped_predictions = predictions.groupBy("user_id").agg(collect_list(struct("game_id", "rating", "prediction")).alias("predictions"))

    precisions = []
    recalls = []
    similarity_scores = []
    for row in grouped_predictions.collect():
        user_ratings = row["predictions"]

        # Sort user ratings by predicted value
        user_ratings.sort(key=lambda x: x["prediction"], reverse=True)

        # Get the top-k predictions
        top_k = user_ratings[:k]

        # Number of relevant items
        n_rel = sum((rating >= threshold) for (_, rating, _) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((predicted >= threshold) for (_, _, predicted) in top_k)

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((rating >= threshold) and (predicted >= threshold)) for (_, rating, predicted) in top_k)

        # Precision@K: Proportion of recommended items that are relevant
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        precisions.append(precision)
        recalls.append(recall)

        games_ids = [iid for (iid, _, _) in user_ratings[:k]]
        user_diversity_score = intra_list_dissimilarity(games_ids)
        similarity_scores.append(user_diversity_score)
    return np.mean(precisions), np.mean(recalls), np.mean(similarity_scores)

# Set the K value to change the number of top-k recommendations to evaluate
K=10
precisions, recalls, diversity = precision_recall_diversity_at_k(dfs_pred, k=K, threshold=3)
# Precision and recall can then be averaged over all users

novelty_score = novelty_at_k(dfs_pred, K)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(dfs_pred)
evaluator2 = RegressionEvaluator(metricName='mae', labelCol="rating", predictionCol="prediction")
mae = evaluator2.evaluate(dfs_pred)

print("P@K", precisions)
print("R@K", recalls)
print("RMSE:", rmse)
print("MAE:", mae)
print("Novelty score:", novelty_score)
print("Diversity score:", diversity)
end_time = time.time()
time_overall = end_time - start_time
print("took", time_overall, "seconds")
# Write evaluation results to file
spark.stop()