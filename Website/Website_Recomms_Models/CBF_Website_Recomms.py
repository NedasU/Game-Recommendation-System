import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import time
start = time.time()

# Prepares the data for use
def data_prep():
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
    return df, df2

# Checks if the user SteamID exists and if it doesnt, returns None. If it does, it gets the top predicted games as
#recommendations and for each game, Steam Store API calls are made to get their data. This data is returned

key = ...
# Must have a valid steam account to generate a steam key. Add the key above to use the recommendation system
def get_user_recommendations(steamid):
    try:
        data = requests.get(
            f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={key}&steamid={steamid}").json()
        user_games = [game['appid'] for game in data['response']['games']]
    except:
        return None
    df, df2 = data_prep()

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
        if i == 10:
            break
    recommended_games = zip(df.iloc[top_similar_games_indices]['id'].tolist(), top_similar_games_scores)

    recomms = []
    for gameid, score in recommended_games:
        info = requests.get(f"https://store.steampowered.com/api/appdetails/?appids={gameid}").json()
        image = info[str(gameid)]['data']['header_image']
        name = info[str(gameid)]['data']['name']
        recomms.append({"name": name, "image_url": image, "id": gameid, 'confidence': score * 100})
    return recomms




