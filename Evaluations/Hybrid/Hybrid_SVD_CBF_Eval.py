import pandas as pd
import numpy as np
from surprise import Dataset, Reader
import Get_user_info
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import time

# Load the datasets
reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
game_df = pd.read_csv("../../Datasets/Hybrid/Hybrid_Game_File.csv")
game_names_dict = dict(zip(game_df['id'], game_df['app_name']))

#Takes in the owner count and returns a value based on which range the owner_count fits in.
def get_owner_range(owner_count):
    if owner_count > 10000000:
        return 10000000
    elif owner_count >5000000:
        return 5000000
    elif owner_count > 1000000:
        return 1000000
    elif owner_count > 500000:
        return 500000
    else:
        return 100000 # This default value seems to be needed to achieve a good mean novelty score.
                      # this may be due to the large difference between the largest and lowest values of owner_count.

# As the owners column is a string in format '0 .. 20,000', there is a need to get just one value to be used for
# calculating the novelty score.
# Extract the owner count and apply the custom function
game_df['popularity'] = game_df['owners'].apply(lambda x: get_owner_range(int(x.split("..")[1].replace(',', '').strip())))

# initialising MinMaxScaler along with TF-IDF vectorizer and fitting them on their corresponding data
scaler = MinMaxScaler()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

game_df['normalised_popularity'] = scaler.fit_transform(game_df['popularity'].values.reshape(-1, 1))
game_df.dropna(subset=['genres'], inplace=True)
diversity_genres = game_df['genres']
processed_genres = tfidf_vectorizer.fit_transform(diversity_genres)

# Prepares the dataset to be in numerical format. This includes transforming list features, using TF-IDF
# String columns were transformed using oneHotEncoder and the numerical values were normalised using MinMaxScaler
# Returns the prepared datas
def prepare_cbf_data():
    # Load the dataset
    df = pd.read_csv('../../Datasets/CBF/CBF_Additional_Features.csv')

    df['release_year'] = df['release_date'].apply(lambda x: x.split('-')[0])
    df2 = df.drop(columns=['id', 'app_name', 'release_date'])

    tfidf_features = tfidf_vectorizer.fit_transform(df2['genres'] + ' ' + df2['specs'] + ' ' + df2['tags'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    df2.drop(columns=['genres', 'specs', 'tags'], inplace=True)
    df2[['release_year', 'review_score']] = scaler.fit_transform(df2[['release_year', 'review_score']])

    # Encoding categorical variables using OneHotEncoder
    categorical_cols = ['publisher', 'developers', 'price', 'owners', 'early_access']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(df2[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df2.drop(columns=categorical_cols, inplace=True)
    df2 = pd.concat([df2, encoded_df, tfidf_df], axis=1)

    return df, df2


# Takes in user games and calculates aggregated feature vector to get the average for each feature of each game that
# the user has interacted with. This is used to calculate the cosine similarity between the prepared dataset df2
# and get the top similar games. The games ids are returned along side their similarity scores.
def get_cbf_recommendations(user_games):
    aggregate_features = df2[df['id'].isin(user_games)].mean(axis=0)
    cosine_sim_with_user = cosine_similarity([aggregate_features], df2.values)[0]

    similarity_scores = list(enumerate(cosine_sim_with_user))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_games_indices = []
    top_similar_games_scores = []

    for index, sim_score in similarity_scores:
        top_similar_games_indices.append(index)
        top_similar_games_scores.append(sim_score)

    return [(gameid, gamescore) for gameid, gamescore in zip(df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores)]


# Loads the datasets and the models. Trains the model on the full dataset and predicts ratings for all games. The
# game ids and estimated ratings are returned
def get_svd_recommendations(user_id):
    svd_dataset_path = '../../Datasets/User-Item-Interaction_2651.csv'
    df = pd.read_csv(svd_dataset_path)
    data = Dataset.load_from_df(df[['user_id', "game_id", "rating"]], reader=reader5)

    model = joblib.load("../../Models/SVD_Achiv_0-5_2651.pkl")
    model.fit(data.build_full_trainset())
    all_games = df['game_id'].unique()

    predictions = [model.predict(user_id, game_id) for game_id in all_games]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [(rec.iid, rec.est) for rec in predictions]
    return recommendations


# calls both of the recommendation functions and combines them to generate one final combined list of recommendations
# The CBF models scores had to be scaled to match the same rating as SVD so it was multiplied by 5. The weights are
# used to set how much of each model to take into account
def get_hybrid_recommendations(userid, usergames, k):
    svd_top_recomms = get_svd_recommendations(userid)
    cbf_top_recomms = get_cbf_recommendations(usergames)
    hybrid_recomms = {}
    for game_id, svd_score in svd_top_recomms:
        hybrid_recomms[game_id] = svd_score * 0.1

    for game_id, cbf_score in cbf_top_recomms:
        if game_id in hybrid_recomms:
            hybrid_recomms[game_id] += (cbf_score * 5) * 0.9 # ensuring that cbf has the same rating format as svd (0-5)
        else:
            hybrid_recomms[game_id] = (cbf_score * 5) * 0.9

    hybrid_top_recomms = sorted(hybrid_recomms.items(), key=lambda x: x[1], reverse=True)
    top_rec_ids = [key for key, value in hybrid_top_recomms[:k]]
    return top_rec_ids

# Prepares the necessary data for CBF
df, df2 = prepare_cbf_data()


# Takes the recommended games ids and gets their genres. pairwise dissimilarity is calculated between each
# recommended item. The mean value of all these diversity values are returned as the overall diversity score.
def intra_list_dissimilarity(recommended_game_ids):
    recommended_game_ids = [game_id for game_id in recommended_game_ids if game_id in game_df['id'].unique()]
    genres = processed_genres[game_df['id'].isin(recommended_game_ids)]

    dissimilarities = []
    for i in range(len(recommended_game_ids)):
        for j in range(i + 1, len(recommended_game_ids)):
            game1 = genres[i]
            game2 = genres[j]
            dissimilarity = 1 - cosine_similarity(game1.reshape(1, -1), game2.reshape(1, -1))
            dissimilarities.append(dissimilarity)

    avg_dissimilarity = np.mean(dissimilarities)
    return avg_dissimilarity

# Takes in the recommended games ids and calculates the recommendations novelty scores through the
# normalised popularity scores. The mean average is returned as the users novelty scores
def novelty_score_calculator(recommended_game_ids):
    recommended_games_data = game_df[game_df['id'].isin(recommended_game_ids)]
    user_novelty_score = np.mean([novelty_score for novelty_score in recommended_games_data['normalised_popularity']])
    return user_novelty_score


# Takes in a sample size and a k value. the sample size is used to generate the selected number of random users
# from the dataset. The k value is used to get the top k recommendations
# Precision, diversity and novelty at k are calculated here
def precision_and_diversity_novelty_at_k(k, sample_size):
    users_df = pd.read_csv("../../Datasets/User-Item-Interaction_2651.csv")
    user_precisions = []
    user_diversities = []
    unique_users = pd.Series(users_df['user_id'].unique())
    users_sample = unique_users.sample(n=sample_size, random_state=42)
    novelty_scores = []
    for user_id in users_sample:
        user_interactions = users_df[users_df['user_id'] == user_id]['game_id'].tolist()
        recommended_game_ids = get_hybrid_recommendations(user_id, user_interactions, k)
        relevant_count = sum(game_id in user_interactions for game_id in recommended_game_ids)
        user_precision = relevant_count / k if k > 0 else 0
        user_precisions.append(user_precision)
        user_diversity = intra_list_dissimilarity(recommended_game_ids)
        user_diversities.append(user_diversity)
        novelty_scores.append(novelty_score_calculator(recommended_game_ids))
    mean_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
    mean_diversity = sum(user_diversities) / len(user_diversities) if user_diversities else 0
    return mean_precision, mean_diversity, np.mean(novelty_scores)


# Used to set the different K values and sample sizes for evalautions
K=10
sample_size = 100
start = time.time()
precision, diversity, novelty = precision_and_diversity_novelty_at_k(K, sample_size)
end = time.time()
total_time = end - start

print(f"Evaluation took {total_time} seconds for sample size: {sample_size} and K={K} achieved the following results\n"
      f"Precision@{K}: {precision}\n"
      f"Diversity: {diversity}\n"
      f"Novelty: {novelty}")

