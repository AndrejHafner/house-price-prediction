import numpy as np
import time
import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from utils import *
from config import *


models = ["svm","rf","xgb","lgbm"]

def train(model, x, y, params = None, print_progress = True):

    if model not in models:
        print("Unknown model - exiting.")
        return

    start_time = time.time()
    reg_model = get_model(model, params)
    reg_model.fit(x,y)

    if print_progress: print(f"Training {model} took {round(time.time() - start_time)} seconds.")
    return reg_model

def test(model, x, df_test):
    predictions = model.predict(x)

    if isinstance(model, SVR):
        save_submission(SVM_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, RandomForestRegressor):
        save_submission(RANDOM_FOREST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, xgb.XGBRegressor):
        save_submission(XGBOOST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, lgbm.LGBMRegressor):
        save_submission(LIGHTGBM_SUBMISSION_DIR, df_test, predictions)

def cv(model, x, y, cv = 10):
    scorers = {
        "rmse": make_scorer(mean_squared_error, squared=False)
    }
    cv_result = cross_validate(model, x, y, cv=cv, n_jobs=8, scoring=scorers)
    rmse = float(np.mean(cv_result["test_rmse"]))
    print(f"Cross validation complete. Folds: {cv}, Mean RMSE: { round(rmse, 2) }.")


def get_model(model, params = None):
    if model not in models:
        print("Unknown model - exiting.")
        return

    if model is "svm":
        if params is None: params = SVM_CONFIG
        return SVR(**params)
    elif model is "rf":
        if params is None: params = RANDOM_FOREST_CONFIG
        return RandomForestRegressor(**params)
    elif model is "xgb":
        if params is None: params = XGBOOST_CONFIG
        return xgb.XGBRegressor(**params)
    elif model is "lgbm":
        if params is None: params = LIGHTGBM_CONFIG
        return lgbm.LGBMRegressor(**params)

