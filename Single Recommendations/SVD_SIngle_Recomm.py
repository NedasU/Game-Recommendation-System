# To run this program click run. You'll be promted for an input of your steamID. This ID will be used to make a call
# to get the user data in a correct format alongside the whole dataset.

from surprise import Dataset, Reader
import Get_user_info
import joblib
import time
start = time.time()

# Load your data from a pandas DataFrame or CSV file
# Ensure that your data has columns: 'user_id', 'game_id', 'rating'
# Load data from CSV file (example)
reader5 = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))

# To get the playtime versions, edit the second parameter to false of the get_user_games function and also change the
# loaded model.
user_id = input("Enter your SteamID: ")
df = Get_user_info.get_user_games(user_id, True)
data = Dataset.load_from_df(df[['user_id', "game_id", "rating"]], reader=reader5)

model = joblib.load("../Models/SVD_Achiv_0-5_2651.pkl")

#Fits the model and makes predicted ratings to unseen games. Prints the top predicted rating games as recommendations
model.fit(data.build_full_trainset())

user_id = 10000
all_games = df['game_id'].unique()
user_data = df[df['user_id'] == user_id]

# Get the unique game IDs for that user
seen_games = user_data['game_id'].unique()
unseen_games = set(all_games) - set(seen_games)

predictions = [model.predict(user_id, game_id) for game_id in unseen_games]
predictions.sort(key=lambda x: x.est, reverse=True)
for rec in predictions[:10]:
    print(f"ID: {rec.iid}, estimated rating: {rec.est}")

end = time.time()
total_time = end - start
print(f"Operation Took: {total_time} seconds")