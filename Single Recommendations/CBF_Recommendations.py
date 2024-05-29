# Click run to run this program. Edit the K value to change the number of recommendations to return

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import time
start = time.time()

#Loads the datasets and performs preprocessing to ensure all data is in numerical format
df = pd.read_csv('../Datasets/CBF/CBF_Additional_Features.csv')
df['release_year'] = df['release_date'].apply(lambda x: x.split('-')[0])
df2 = df.drop(columns=['id', 'app_name', 'release_date'])
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(df2['genres'] + ' ' + df2['specs'] + ' ' + df2['tags'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

df2.drop(columns=['genres', 'specs', 'tags'], inplace=True)

scaler = MinMaxScaler()
df2[['release_year', 'review_score']] = scaler.fit_transform(df2[['release_year', 'review_score']])
categorical_cols = ['publisher', 'developers', 'price', 'early_access', 'owners']
encoder = OneHotEncoder(sparse_output=False)
encoded_cols = encoder.fit_transform(df2[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

df2.drop(columns=categorical_cols, inplace=True)

df2 = pd.concat([df2, encoded_df, tfidf_df], axis=1)

# Takes in user games and calculates aggregated feature vector to get the average for each feature of each game that
# the user has interacted with. This is used to calculate the cosine similarity between the prepared dataset df2
# and get the top similar games. The top-k games ids are returned along side their similarity scores.
def get_user_recommendations(user_games, K=10):
    aggregate_features = df2[df['id'].isin(user_games)].mean(axis=0)
    cosine_sim_with_user = cosine_similarity([aggregate_features], df2.values)[0]

    similarity_scores = list(enumerate(cosine_sim_with_user))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_games_indices = []
    top_similar_games_scores = []
    i = 0
    for index, sim_score in similarity_scores:
        if df.iloc[index]['id'] not in user_games:
            top_similar_games_indices.append(index)
            top_similar_games_scores.append(sim_score)
            i += 1
        if i == K:
            break
    return [df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores]


# Gets user games by making a call to the Steam Web API to get all the users games.

key = ...
# Must have  valid Steam account to generate  a steam key. Enter the steam key above to use the recommendation system
def get_user_games(userid, K):
    data = requests.get(f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={key}&steamid={userid}").json()
    return get_user_recommendations([game['appid'] for game in data['response']['games']], K)


# Prompts for the user steam id anc makes a call to get the recommendations and displays them
user_id = input("Enter your Steam ID: ")
recommended_games, similarity_scores = get_user_games(user_id, 10)
print("Recommended Games:")
end = time.time()
for n in range(len(recommended_games)):
    print(recommended_games[n], similarity_scores[n])

time_overall = end - start

print(f"Time Taken: {time_overall}")