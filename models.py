import numpy as np
import time
import xgboost as xgb
import lightgbm as lgbm
from mlxtend.regressor import StackingCVRegressor

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.dummy import DummyRegressor
from utils import *
from config import *


models = ["svm", "rf", "xgb", "lgbm", "ridge_lr", "lasso_lr", "elasticnet_lr", "lr", "avg"]

def train(model, x, y, print_progress = True, **kwargs):

    if model not in models:
        print("Unknown model - exiting.")
        return

    start_time = time.time()
    reg_model = get_model(model, **kwargs)
    reg_model.fit(x, y.values.ravel())

    if print_progress: print(f"Training {model} took {round(time.time() - start_time)} seconds.")
    return reg_model

def save_submission(model, predictions, df_test):
    if isinstance(model, Pipeline):
        if isinstance(model.steps[1][1], LinearRegression):
            save_submission_file(LINEAR_REGRESSION_SUBMISSION_DIR, df_test, predictions)
        if isinstance(model.steps[1][1], RidgeCV):
            save_submission_file(RIDGE_REGRESSION_SUBMISSION_DIR, df_test, predictions)
        if isinstance(model.steps[1][1], LassoCV):
            save_submission_file(LASSO_REGRESSION_SUBMISSION_DIR, df_test, predictions)
        if isinstance(model.steps[1][1], ElasticNetCV):
            save_submission_file(ELASTICNET_REGRESSION_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, RandomForestRegressor):
        save_submission_file(RANDOM_FOREST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, xgb.XGBRegressor):
        save_submission_file(XGBOOST_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, lgbm.LGBMRegressor):
        save_submission_file(LIGHTGBM_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, LinearRegression):
        save_submission_file(LINEAR_REGRESSION_SUBMISSION_DIR, df_test, predictions)
    elif isinstance(model, DummyRegressor):
        save_submission_file(DUMMY_REGRESSOR_AVG_SUBMISSION_DIR, df_test, predictions)

def rmse_exp(y, y_pred):
    y_exp = np.exp(y)
    y_pred_exp = np.exp(y_pred)
    return np.sqrt((1 / len(y_exp)) * np.sum((y_pred_exp - y_exp)**2))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv(model, x, y, k = 10, model_name = ""):
    scorers = {
        "rmsle": make_scorer(rmsle, greater_is_better=True),
        "rmse": make_scorer(rmse_exp, greater_is_better=True)
    }

    cv_result = cross_validate(model, x, y.values.ravel(), cv=k, n_jobs=8, scoring=scorers)
    rmsle_score = float(np.mean(cv_result["test_rmsle"]))
    rmse_score = float(np.mean(cv_result["test_rmse"]))
    print(f"Cross validation complete - { model_name }. Folds: { k }, RMSE: { round(rmse_score, 4) }, RMSLE: {round(rmsle_score, 4)}.")


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
    elif model == "ridge_lr":
        if len(kwargs) == 0: kwargs = RIDGE_LR_CONFIG
        return make_pipeline(RobustScaler(), RidgeCV(**kwargs))
    elif model == "lasso_lr":
        if len(kwargs) == 0: kwargs = LASSO_LR_CONFIG
        return make_pipeline(RobustScaler(), LassoCV(**kwargs))
    elif model == "elasticnet_lr":
        if len(kwargs) == 0: kwargs = ELASTICNET_LR_CONFIG
        return make_pipeline(RobustScaler(), ElasticNetCV(**kwargs))
    elif model == "stacked":
        if len(kwargs) == 0: kwargs = STACKED_CONFIG

        regressors = [
            get_model("xgb"),
            get_model("svm"),
            get_model("rf"),
            get_model("lgbm")
        ]
        kwargs["regressors"] = regressors
        kwargs["meta_regressor"] = regressors[0]

        return StackingCVRegressor(**kwargs)
    elif model == "lr":
        if len(kwargs) == 0: kwargs = LR_CONFIG
        return make_pipeline(LinearRegression(**kwargs))
    elif model == "avg":
        if len(kwargs) == 0: kwargs = AVG_CONFIG
        return DummyRegressor(**kwargs)

