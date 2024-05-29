# To run, click the run button. To use grid search with the playtime dataset,
# change the dataset path to include the playtime version

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import collect_list, struct, col, desc, array_contains
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time
import os
import sys
import Get_user_info


# Initialises the spark session
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder \
    .appName("ALS Recommender") \
    .config("spark.driver.memory", "6g")\
    .getOrCreate()

dataset_path = "../Datasets/User-Item-Interaction_2651.csv"

df = spark.read.csv(dataset_path, schema='user_id int, game_id int, rating int', header=True)

dfs_train, dfs_test = df.randomSplit([0.8, 0.2], seed=42)

USER_COL = "user_id"
ITEM_COL = "game_id"
RATING_COL = "rating"
PREDICTION_COL = "prediction"
start_time = time.time()

als = ALS(
    userCol=USER_COL,
    itemCol=ITEM_COL,
    ratingCol=RATING_COL,
    coldStartStrategy='drop',
    nonnegative=True,
    implicitPrefs=False,
    seed=42,
)

## The below are the best parameters for each dataset (Playtime and Achievements)
# Best parameters: 0-5 2651 achievement:
# RegParam: 0.1
# MaxIter: 20
# Rank: 20
# took 1149.1347150802612 seconds

# Best parameters: 0-5-2651 playtime:
# RegParam: 0.1
# MaxIter: 20
# Rank: 30
# took 1122.4485092163086 seconds

# Parameters to test
param_grid = (
    ParamGridBuilder()
    .addGrid(als.rank, [5, 10, 15, 20, 30])
    .addGrid(als.regParam, [0.01, 0.1, 1.0, 10])
    .addGrid(als.maxIter, [10, 15, 20])
    .addGrid(als.seed, [42])
    .build()
)

# evaluator to be used as the goal of reducing in regards to using the cross validator
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)

print("Number of models to be tested:", len(param_grid))


# Crossvalidator that sets the number of folds to be used in order to find the best parameters
cv = CrossValidator(
    estimator=als,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)

print("Fitting model to the data for cross validation and hyper parameter tuning")
model = cv.fit(dfs_train)

# For some reason I could not save the best models for which I had to print the best parameters instead.
# Prints the best parameters for each model
print("Best parameters:")
print("RegParam:", model.bestModel._java_obj.parent().getRegParam())
print("MaxIter:", model.bestModel._java_obj.parent().getMaxIter())
print("Rank:", model.bestModel._java_obj.parent().getRank())

spark.stop()