from surprise import Dataset, Reader
import Get_user_info
import joblib
import requests

# Checks if the user Steam ID is valid and the account is not private. If it is, then None is returned to show an error.
# If not, the loaded model * Achievement in this case* is used to fit the trainset on and make predictions for the
# chosen user the recommended games information is gathered through the Steam Store API before returning the data.
def SVD_Recommend(id):
    reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0, 5))

    df = Get_user_info.get_user_games(id,True)
    if df is None:
        return None

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
    recomms = []
    for rec in predictions[:11]:
        info = requests.get(f"https://store.steampowered.com/api/appdetails/?appids={rec.iid}").json()
        try:
            image = info[str(rec.iid)]['data']['header_image']
            name = info[str(rec.iid)]['data']['name']
        except:
            continue
    recomms.append({"name": name, "image_url": image, "id":rec.iid, "confidence":(rec.est /5)*100})
    return recomms
