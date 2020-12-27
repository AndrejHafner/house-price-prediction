# Model parameters to be used during training

SVM_CONFIG = dict(
    C=20,
    epsilon= 0.008,
    gamma=0.0003
)

RANDOM_FOREST_CONFIG = dict(
    n_estimators=1200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features=None,
    oob_score=True,
    random_state=42
)

XGBOOST_CONFIG = dict(
    learning_rate=0.01,
    n_estimators=3460,
    max_depth=3,
    min_child_weight=0,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:squarederror',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006
)

LIGHTGBM_CONFIG = dict(
    objective='regression',
    num_leaves=4,
    learning_rate=0.01,
    n_estimators=5000,
    max_bin=200,
    bagging_fraction=0.75,
    bagging_freq=5,
    bagging_seed=7,
    feature_fraction=0.2,
    feature_fraction_seed=7,
    verbose=-1,
)

LR_CONFIG = {

}

AVG_CONFIG = {

}