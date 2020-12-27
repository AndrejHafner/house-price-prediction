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
    n_estimators=6000,
    max_depth=4,
    min_child_weight=0,
    gamma=0.6,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='reg:linear',
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006,
    random_state=42
)

LIGHTGBM_CONFIG = dict(
    objective='regression',
   num_leaves=6,
   learning_rate=0.01,
   n_estimators=7000,
   max_bin=200,
   bagging_fraction=0.8,
   bagging_freq=4,
   bagging_seed=8,
   feature_fraction=0.2,
   feature_fraction_seed=8,
   min_sum_hessian_in_leaf = 11,
   verbose=-1,
   random_state=42
)

LR_CONFIG = dict(
)

AVG_CONFIG = dict(

)

RIDGE_LR_CONFIG = dict(
    alphas=[1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100],
    cv=10
)

LASSO_LR_CONFIG = dict(
    max_iter=1e7,
    alphas=[5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008],
    random_state=42,
    cv=10
)

ELASTICNET_LR_CONFIG = dict(
    max_iter=1e7,
    alphas=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007],
    cv=10,
    l1_ratio=[0.8, 0.85, 0.9, 0.95, 0.99, 1]
)

STACKED_CONFIG = dict(
    use_features_in_secondary=True
)
