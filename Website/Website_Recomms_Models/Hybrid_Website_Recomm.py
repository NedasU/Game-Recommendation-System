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

# Loads the Hybrid Dataset
reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
game_df = pd.read_csv("../Datasets/Hybrid/Hybrid_Game_File.csv")
game_names_dict = dict(zip(game_df['id'], game_df['app_name']))


# Prepares the data for use for CBF
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

# Checks if the user SteamID exists and if it doesnt, returns None. If it does, it gets the top predicted games as
#recommendations and returns them.

key = ...
# Must have a valid steam account and generate a steam key to access the steam dataset. Add the key in the above variable.
def get_cbf_recommendations(userid):
    try:
        data = requests.get(f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={key}&steamid={userid}").json()
        user_games = [game['appid'] for game in data['response']['games']]
    except:
        return None
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

    return [(game_id, score) for game_id, score in zip(df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores)]


# Gets user information and loads the chosen model *Achievment in this case*
# Fits the training set to the model and predicts ratings for the unseen games.
# Returns the top games
def get_svd_recommendations(user_id):
    df = Get_user_info.get_user_games(user_id,True)

    data = Dataset.load_from_df(df[['user_id', "game_id", "rating"]], reader=reader5)

    model = joblib.load("../Models/SVD_Achiv_0-5_2651.pkl")
    model.fit(data.build_full_trainset())

    user_id = 10000
    all_games = df['game_id'].unique()
    user_data = df[df['user_id'] == user_id]

    # Get the unique game IDs for that user
    seen_games = user_data['game_id'].unique()
    unseen_games = set(all_games) - set(seen_games)

    predictions = [model.predict(user_id, game_id) for game_id in unseen_games]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [(rec.iid, rec.est) for rec in predictions]
    return recommendations


# Checks to see if User SteamID is available through the first call to the get_cbf_recommendations.
# SVD models scores are divided by 5 to make sure that both of the scores are in similar format (0-1 in this case)
# Iterates through each recommendation for both models and applies weights to get scores. These recommendations are
# combined and the highest scored games get returned
def get_hybrid_recommendations(userid, k):
    cbf_top_recomms = get_cbf_recommendations(userid)
    if cbf_top_recomms is None:
        return None
    svd_top_recomms = get_svd_recommendations(userid)

    hybrid_recomms = {}
    for game_id, svd_score in svd_top_recomms:
        hybrid_recomms[game_id] = (svd_score/5) * 0.1

    for game_id, cbf_score in cbf_top_recomms:
        if game_id in hybrid_recomms:
            hybrid_recomms[game_id] += (cbf_score) * 0.9 # ensuring that cbf has the same rating format as svd (0-5)
        else:
            hybrid_recomms[game_id] = (cbf_score) * 0.9

    hybrid_top_recomms = sorted(hybrid_recomms.items(), key=lambda x: x[1], reverse=True)[:k]
    return hybrid_top_recomms

df, df2 = prepare_cbf_data()

# Takes in the steam id and returns None if SteamId is private or wrong. Gets the information for each recommendation
# by making calls to the Steam Store API and returns the information.
def Recommendations(steamid):
    recomms = []
    recommendations = get_hybrid_recommendations(steamid, 10)
    if recommendations is None:
        return None
    for gameid, score in recommendations:
        info = requests.get(f"https://store.steampowered.com/api/appdetails/?appids={gameid}").json()
        image = info[str(gameid)]['data']['header_image']
        name = info[str(gameid)]['data']['name']
        recomms.append({"name": name, "image_url": image, "id": gameid, 'confidence':score*100})
    return recomms
