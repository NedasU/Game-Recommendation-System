

import pandas as pd
from surprise import Dataset, Reader
import Get_user_info
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import time
start = time.time()
reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
game_df = pd.read_csv("../Datasets/Hybrid/Hybrid_Game_File.csv")
game_names_dict = dict(zip(game_df['id'], game_df['app_name']))


# Prepares the dataset to be used for the CBF model
def prepare_cbf_data():
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    df = pd.read_csv('../Datasets/CBF/CBF_Additional_Features.csv')
    df['release_year'] = df['release_date'].apply(lambda x: x.split('-')[0])
    df2 = df.drop(columns=['id', 'app_name', 'release_date'])

    tfidf_features = tfidf_vectorizer.fit_transform(df2['genres'] + ' ' + df2['specs'] + ' ' + df2['tags'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    df2.drop(columns=['genres', 'specs', 'tags'], inplace=True)

    scaler = MinMaxScaler()
    df2[['release_year', 'review_score']] = scaler.fit_transform(df2[['release_year', 'review_score']])

    categorical_cols = ['publisher', 'developers', 'price', 'owners', 'early_access']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(df2[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))

    df2.drop(columns=categorical_cols, inplace=True)

    df2 = pd.concat([df2, encoded_df, tfidf_df], axis=1)
    return df, df2


# Takes in user games and calculates aggregated feature vector to get the average for each feature of each game that
# the user has interacted with. This is used to calculate the cosine similarity between the prepared dataset df2
# and get the top similar games. The top-k games ids are returned along side their similarity scores.
def get_cbf_recommendations(user_games):
    df, df2 = prepare_cbf_data()
    aggregate_features = df2[df['id'].isin(user_games)].mean(axis=0)
    cosine_sim_with_user = cosine_similarity([aggregate_features], df2.values)[0]

    similarity_scores = list(enumerate(cosine_sim_with_user))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_games_indices = []
    top_similar_games_scores = []

    for index, sim_score in similarity_scores:
        if df.iloc[index]['id'] not in user_games:
            top_similar_games_indices.append(index)
            top_similar_games_scores.append(sim_score)

    return [(gameid, gamescore) for gameid, gamescore in zip(df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores)]


# Gets the user data and explicit forms of ratings. Loads the models *Achievement version in this case*
# predicts ratings for unseen games and returns the top recommendations
def get_svd_recommendations(user_id):
    df = Get_user_info.get_user_games(user_id, True)
    data = Dataset.load_from_df(df[['user_id', "game_id", "rating"]], reader=reader5)

    model = joblib.load("../Models/SVD_Achiv_0-5_2651.pkl")
    model.fit(data.build_full_trainset())
    all_games = df['game_id'].unique()
    userid = 10000
    user_data = df[df['user_id'] == userid]
    seen_games = user_data['game_id'].unique()
    unseen_games = set(all_games) - set(seen_games)
    predictions = [model.predict(user_id, game_id) for game_id in unseen_games]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [(rec.iid, rec.est) for rec in predictions]
    return recommendations

# Makes a call to an API to get user games. Normalises the scores of CBF model to make a rating system of 0-5 and to
# match the format of SVD. Also, for each recommendation, their score is multiplied by the corresponding weight
# of each model. These recommendations are combined together and the top scored recommendations are returned.

key = ...
# Must have  valid Steam account to generate  a steam key. Enter the steam key above to use the recommendation system
def get_hybrid_recommendations(userid, k):
    userinfo = requests.get(f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={key}"
                             f"&steamid={userid}&include_played_free_games=true").json()
    usergames = [game['appid'] for game in userinfo['response']['games']]
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
    top_rec_ids = [(key, value) for key, value in hybrid_top_recomms[:k]]
    return top_rec_ids


# Prompts the user for their steam id and makes a call to a function. Prints out the ids and scores of
# each game recommended.
user_id = input("Enter your Steam ID: ")
for (id, score) in get_hybrid_recommendations(user_id, 10):
    print(f"GameID: {id}, Score: {score}")

end = time.time()
total_time = end - start
print(f"Took time: {total_time}")

