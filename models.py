import numpy as np
import time
import xgboost as xgb
import lightgbm as lgbm

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from utils import *
from config import *


models = ["svm", "xgb","lgbm", "lr", "avg", "rf"]

def train(model, x, y, print_progress = True, **kwargs):

    if model not in models:
        print("Unknown model - exiting.")
        return

    start_time = time.time()
    reg_model = get_model(model, **kwargs)
    reg_model.fit(x,y)

    if print_progress: print(f"Training {model} took {round(time.time() - start_time)} seconds.")
    return reg_model

def save_submission(model, predictions, df_test):
    if isinstance(model, Pipeline):
        save_submission_file(SVM_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, RandomForestRegressor):
        save_submission_file(RANDOM_FOREST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, xgb.XGBRegressor):
        save_submission_file(XGBOOST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, lgbm.LGBMRegressor):
        save_submission_file(LIGHTGBM_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, LinearRegression):
        save_submission_file(LIENAR_REGRESSION_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, DummyRegressor):
        save_submission_file(DUMMY_REGRESSOR_AVG_SUBMISSION_DIR, df_test, predictions)

def cv(model, x, y, k = 10, model_name = ""):
    scorers = {
        "rmse": make_scorer(mean_squared_error, squared=False)
    }

    cv_result = cross_validate(model, x, y.values.ravel(), cv=k, n_jobs=8, scoring=scorers)
    rmse = float(np.mean(cv_result["test_rmse"]))
    print(f"Cross validation complete - { model_name }. Folds: { k }, Mean RMSE: { round(rmse, 3) }.")


def cv_all_models(x,y, **kwargs):
    for model_name in models:
        cv(get_model(model_name), x, y, **kwargs, model_name=model_name)

def get_model(model, **kwargs):
    if model not in models:
        print("Unknown model - exiting.")
        return

    if model == "svm":
        if len(kwargs) == 0: kwargs = SVM_CONFIG
        return make_pipeline(RobustScaler(), SVR(**kwargs))
    elif model == "rf":
        if len(kwargs) == 0: kwargs = RANDOM_FOREST_CONFIG
        return RandomForestRegressor(**kwargs)
    elif model == "xgb":
        if len(kwargs) == 0: kwargs = XGBOOST_CONFIG
        return xgb.XGBRegressor(**kwargs)
    elif model == "lgbm":
        if len(kwargs) == 0: kwargs = LIGHTGBM_CONFIG
        return lgbm.LGBMRegressor(**kwargs)
    elif model == "lr":
        if len(kwargs) == 0: kwargs = LR_CONFIG
        return LinearRegression(**kwargs)
    elif model == "avg":
        if len(kwargs) == 0: kwargs = AVG_CONFIG
        return DummyRegressor(**kwargs)

