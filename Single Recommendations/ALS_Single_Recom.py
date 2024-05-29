# To run this program click run. You'll be promted for an input of your steamID. This ID will be used to make a call
# to get the user data in a correct format alongside the whole dataset.

from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import os
import sys
import Get_user_info
import time

#Initialise the spark session
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
start = time.time()
spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.driver.memory", "6g")\
    .getOrCreate()

# Get user info. Additionally, If you wish to use playtime as the version, edit the second parameter to False
user_id = input("Enter your SteamID: ")
before_df = Get_user_info.get_user_games(user_id, True)
df = spark.createDataFrame(before_df)
# df = spark.read.csv(updated_dataset_path, schema='user_id int, game_id int, rating int', header=True)
dfs_train, dfs_test = df.randomSplit([0.8, 0.2], seed=42)

USER_COL = "user_id"
ITEM_COL = "game_id"
RATING_COL = "rating"
PREDICTION_COL = "prediction"

## These are the two models with different parameters after hypertuning
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

## Best parameters: 0-5-2651_playtime:
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

#Fits the model to the test data
model = als.fit(dfs_train)

# Gets the recommendations for the selected user id for in this case would be the SteamID entered.
selected_user_id = 10000
user_df = dfs_train.filter(col(als.getUserCol()) == selected_user_id).select(als.getUserCol()).distinct()
dfs_rec_subset = model.recommendForUserSubset(user_df, 10)

# Prints out all the recommendations and their corresponding ratings
for row in dfs_rec_subset.collect():
    user_id = row["user_id"]
    recommendations = row["recommendations"]
    print(f"User ID: {user_id}")
    for recommendation in recommendations:
        game_id = recommendation["game_id"]
        rating = recommendation["rating"]
        print(f"  Game ID: {game_id}, Rating: {rating}")
end = time.time()
total_time = end - start
print(f"Took {total_time} seconds")
spark.stop()
