# To run, click the run button. To use grid search with the playtime dataset,
# change the dataset path to include the playtime version

from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV
import joblib


# Load data from CSV file (example)
reader = Reader(line_format='user item rating', sep=',', skip_lines=1, rating_scale=(0,5))
data = Dataset.load_from_file("../Datasets/User-Item-Interaction_2651.csv", reader)

# Defines the params to test
param_grid = {'n_factors': [10, 20, 50],
              'n_epochs': [20, 50, 100],
              'lr_bi': [0.005, 0.01, 0.02],
              'reg_bi': [0.02, 0.1, 0.2],
              'random_state': [42]}

# Initiates the grid search cv and the number of folds and the measure.
grid_search = GridSearchCV(SVD, param_grid=param_grid, measures=['rmse'], cv=5, n_jobs=-1, joblib_verbose=2)
grid_search.fit(data)

# Save the best model
best_model = grid_search.best_estimator['rmse']

# Saves the best model
joblib.dump(best_model, f'Models/SVD/SVD_Achiv_0-5.pkl')

