# This code evaluates the performance of the SVD Model. To use this file, you only need to run the file.
# Changing the value of K can change the number of top-k recommendations are evaluated.
# Additionally, to change the selected model that needs to be evaluated, the dataset and the model need to change:
# The dataset should be this for playtime: "../../Datasets/playtime versions/User-Item-Interaction_2651.csv"
# The model should be this for playtime: "../../Models/SVD_Playtime_0-5_2651.pkl"
# The default version here evaluates the Achievement version already.

from surprise import Dataset, Reader, accuracy
from collections import defaultdict
from surprise.model_selection import KFold
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import time
start = time.time()

#Load Dataset to a pandas dataframe
reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
filename = "../../Datasets/User-Item-Interaction_2651.csv"
df = pd.read_csv(filename)
data = Dataset.load_from_file(filename, reader5)
gameinfo_df = pd.read_csv("../../Datasets/Hybrid/Hybrid_Game_File.csv")

threshold = 3
K = 10

# Initialising and using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
gameinfo_df.dropna(subset=['genres'], inplace=True)
diversity_genres = gameinfo_df['genres']
processed_genres = tfidf_vectorizer.fit_transform(diversity_genres)

#Load the selected model. Additionally, calculate the occurences of games for the
# user-item interaction for use in Novelty calculation
model = joblib.load('../../Models/SVD_Achiv_0-5_2651.pkl')
item_popularity = df['game_id'].value_counts().reset_index()
item_popularity.columns = ['game_id', 'popularity']

# Scale the popularity column for novelty scores.
scaler = MinMaxScaler()
item_popularity['popularity_normalized'] = scaler.fit_transform(item_popularity[['popularity']])


#Calculates the Novelty of the top-k recommendations made to the user. All predicted ratings are only to unseen games.
# The normalised popularity value of each recommendation is taken to calculate the
# mean novelty score of the recommendations
def novelty_at_k(k=10):
    novelty_scores = []
    for user in df['user_id'].unique():
        user_novelty = []
        all_games = df['game_id'].unique()
        user_iter = df[df['user_id'] == user]['game_id'].tolist()
        games_to_rec = set(all_games) - set(user_iter)
        recommendations = [model.predict(user, gameid) for gameid in games_to_rec]
        recommendations.sort(key=lambda x: x.est, reverse=True)
        recs = (rec.iid for rec in recommendations[:k])
        for rec in recs:
            pop_score = item_popularity[item_popularity['game_id'] == rec]['popularity_normalized'].values
            user_novelty.append(pop_score[0] if len(pop_score) > 0 else 0)
        mean_novelty_score = np.mean(user_novelty)
        novelty_scores.append(mean_novelty_score)
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

# Iterates through each user and uses the top-k recommendations to get the diversity scores.
# The mean value of all users is returned.
def diversity_calc(predictions, k):
    user_est_true = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        user_est_true[uid].append((int(iid), est))

    diversity_scores = []
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # Calculate diversity for the recommended items
        rec_items = [iid for (iid,_ ) in user_ratings[:k]]
        user_diversity_score = intra_list_dissimilarity(rec_items)
        diversity_scores.append(user_diversity_score)
    return np.mean(diversity_scores)

# This function is used to calculate the precision and recall at k based on a threshold.
# predicted ratings is taken in and all the top-k recommendations predicted ratings are matched against the threshold
# If the predicted rating is above the threshold it adds to the sum. Returns the precision and recall of each user.
def precision_recall_at_k(predictions, k=30, threshold=3):
    """Return precision and recall at k metrics for each user"""

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

# K-fold evaluation to get an averaged evaluation score of each metric. This makes a more accurate representation of the
# performance of the model
n_splits = 5
kf = KFold(n_splits=n_splits,random_state=42)
similarity_scores = []
p_scores = 0
r_scores = 0

for trainset, testset in kf.split(data):
    # model.fit(trainset)
    predictions = model.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=K, threshold=threshold)
    # Precision and recall can then be averaged over all users
    p_scores += sum(prec for prec in precisions.values()) / len(precisions)
    r_scores += sum(rec for rec in recalls.values()) / len(recalls)


overall_novelty = novelty_at_k(K)
diversity = diversity_calc(predictions, K)
end = time.time()
combined_time = end - start
print("Results\n")
print("P@K", p_scores/n_splits)
print("R@K", r_scores/n_splits)
print("MAE", accuracy.mae(predictions))
print("RMSE", accuracy.rmse(predictions))
print("Diversity score:", diversity)
print(f"Mean Novelty Score: {overall_novelty}")
print("Took", combined_time, "seconds")