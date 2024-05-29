# Click run to run the function. Edit the last function call to change the K value and sample size

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import time

start = time.time()
df = pd.read_csv('../../Datasets/CBF/CBF_Additional_Features.csv')

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
# Additionally, transforms all the data to numerical format for use.
popularity = df['owners'].apply(lambda x: get_owner_range(int(x.split("..")[1].replace(',', '').strip())))
scaler = MinMaxScaler()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
novelty_scores = scaler.fit_transform(popularity.values.reshape(-1, 1))
df['novelty_score'] = novelty_scores
df['release_year'] = df['release_date'].apply(lambda x: x.split('-')[0])
# Dropping unnecessary columns
df2 = df.drop(columns=['id', 'app_name', 'release_date'])
diversity_genres = df2['genres']
processed_genres = tfidf_vectorizer.fit_transform(diversity_genres)

tfidf_features = tfidf_vectorizer.fit_transform(df2['genres'] + ' ' + df2['specs'] + ' ' + df2['tags'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

df2.drop(columns=['genres', 'specs', 'tags'], inplace=True)

df2[['release_year', 'review_score']] = scaler.fit_transform(df2[['release_year', 'review_score']])

categorical_cols = ['publisher', 'developers', 'price', 'early_access', 'owners']
encoder = OneHotEncoder(sparse_output=False)
encoded_cols = encoder.fit_transform(df2[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

df2.drop(columns=categorical_cols, inplace=True)
df2 = pd.concat([df2, encoded_df, tfidf_df], axis=1)

# Takes in user games and calculates aggregated feature vector to get the average for each feature of each game that
# the user has interacted with. This is used to calculate the cosine similarity between the prepared dataset df2
# and get the top similar games.  Additionally, the novelty score is calculated through the normalised 'novelty_score'
# The games ids are returned along side their similarity scores and novelty.
def get_user_recommendations(user_games, K=10):
    aggregate_features = df2[df['id'].isin(user_games)].mean(axis=0)
    cosine_sim_with_user = cosine_similarity([aggregate_features], df2.values)[0]

    similarity_scores = list(enumerate(cosine_sim_with_user))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_games_indices = []
    top_similar_games_scores = []
    top_novelty_scores = []
    for index, sim_score in similarity_scores[:K]:
        top_similar_games_indices.append(index)
        top_similar_games_scores.append(sim_score)
        top_novelty_scores.append(df.iloc[index]['novelty_score'])

    return [df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores, top_novelty_scores]


# Takes the recommended games ids and gets their genres. pairwise dissimilarity is calculated between each
# recommended item. The mean value of all these diversity values are returned as the overall diversity score.
def intra_list_similarity(recommended_game_ids):
    recommended_games_genres = processed_genres[df['id'].isin(recommended_game_ids)]
    dissimilarities = []
    for i in range(len(recommended_game_ids)):
        for j in range(i + 1, len(recommended_game_ids)):
            game1 = recommended_games_genres[i]
            game2 = recommended_games_genres[j]
            dissimilarity = 1 - cosine_similarity(game1.reshape(1, -1), game2.reshape(1, -1))
            dissimilarities.append(dissimilarity)

    avg_dissimilarity = np.mean(dissimilarities)
    return avg_dissimilarity


# Takes in a sample size and a k value. the sample size is used to generate the selected number of random users
# from the dataset. The k value is used to get the top k recommendations
# Precision, diversity and novelty at k are calculated here
def precision_and_diversity_at_k(k, sample_size):
    users_df = pd.read_csv("../../Datasets/User-Item-Interaction_2651.csv")
    user_precisions = []
    user_diversities = []
    user_novelties = []  # Store novelty scores
    unique_users = pd.Series(users_df['user_id'].unique())
    users_sample = unique_users.sample(n=sample_size, random_state=42)
    itera = 0
    for user_id in users_sample:
        itera += 1
        print(itera)
        user_interactions = users_df[users_df['user_id'] == user_id]['game_id'].tolist()
        recommended_game_ids, _, novelty_scores = get_user_recommendations(user_interactions, k)  # Extract novelty scores
        relevant_count = sum(game_id in user_interactions for game_id in recommended_game_ids)
        user_precision = relevant_count / k if k > 0 else 0
        user_precisions.append(user_precision)
        user_diversity = intra_list_similarity(recommended_game_ids)
        user_diversities.append(user_diversity)
        user_novelty = np.mean(novelty_scores)  # Calculate novelty score
        user_novelties.append(user_novelty)
    mean_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
    mean_diversity = sum(user_diversities) / len(user_diversities) if user_diversities else 0
    mean_novelty = sum(user_novelties) / len(user_novelties) if user_novelties else 0  # Calculate mean novelty
    return mean_precision, mean_diversity, mean_novelty


# Change the values within the function call to change the: K-value, and sample size
mean_precision, mean_diversity, mean_novelty = precision_and_diversity_at_k(10, 100)
end = time.time()
print("Mean Average Precision:", mean_precision)
print("Mean Diversity:", mean_diversity)
print("Mean Novelty:", mean_novelty)
total_time = end - start
print("Total time:", total_time)