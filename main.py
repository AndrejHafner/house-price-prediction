import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, boxcox_normmax, probplot, PearsonRConstantInputWarning, PearsonRNearConstantInputWarning
from scipy.special import boxcox1p
from models import train, save_submission, cv_all_models, cv, get_model



def handle_missing_data(df):
    # the data description states that NA refers to typical ('Typ') values
    df['Functional'] = df['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    df['Electrical'] = df['Electrical'].fillna("SBrkr")
    df['KitchenQual'] = df['KitchenQual'].fillna("TA")
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # the data description stats that NA refers to "No Pool"
    df["PoolQC"] = df["PoolQC"].fillna("None")

    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        df[col] = df[col].fillna(0)

    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] = df[col].fillna('None')

    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')

    # Dont know how to update the rest - fill with None
    objects = []
    for i in df.columns:
        if df[i].dtype == object:
            objects.append(i)
    df.update(df[objects].fillna('None'))

    # Same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric.append(i)
    df.update(df[numeric].fillna(0))

    # Some of the non-numeric predictors are stored as numbers -  we convert them into strings
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)

    return df

def fix_skewness(df, skew_thresh = 0.5):
    # Fix features that don't have a normal distribution by applying a Box-Cox transformation
    # Get all the numerical features in our dataset
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric.append(i)

    # Calculate the skew for all the numeric features
    features_skew = df[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

    # Define all the features that have skewness above 0.5 as skewed and apply transformation
    high_skew = features_skew[features_skew > skew_thresh]

    # Apply Box-Cox transformation to skewed features
    for i in high_skew.index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

    return df

def remove_outliars(df):
    # df = df.drop(df[df['Id'] == 1299].index)
    # df = df.drop(df[df['Id'] == 524].index)
    return df

def clean_data(df):
    df = handle_missing_data(df)
    df = remove_outliars(df)
    return df

def feature_engineering(df):

    # Total square footage of the house
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # If the house was remodeled, the date of remodelling is not equal to yea built -> feature that describes if the house was remodeled
    df['Remodeled'] = df['YearBuilt'] != df['YearRemodAdd']

    # Total square footage of the house, where basement is separated by the type of the completion stage -> either type 1 or type 2
    df['TotalSQR'] = df['BsmtFinSF1'] + df['BsmtFinSF2']+ df['1stFlrSF'] + df['2ndFlrSF']

    # Number of all the bathrooms in the house
    df['TotalBathrooms'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

    # Total square footage of all the porches
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']

    # True - false variables based of numerical variables
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    return df

def preprocess_df(df_train, df_test):
    # Remove SalePrice and apply Box-Cox transformation to normalize it
    train_y = np.log(df_train[["SalePrice"]])
    df_train = df_train.drop(['SalePrice'], axis=1)

    # Concat training and testing data to preprocess it together
    df_train["train_set"] = True
    df_test["train_set"] = False

    df = pd.concat([df_train, df_test])
    df = clean_data(df)
    df = fix_skewness(df)
    df = feature_engineering(df)

    # Remove ID - helps nothing as it uniquely identify every entry
    df = df.drop(['Id'], axis=1)

    # Convert categorical data into dummy variables - one hot encoding
    df = pd.get_dummies(df)

    train_x = df[df["train_set"] == True]
    test_x = df[df["train_set"] == False]

    return train_x, train_y, test_x


def main():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    train_x, train_y, test_x = preprocess_df(df_train, df_test)

    # cv_all_models(train_x, train_y)
    # cv(get_model("lr"), train_x, train_y)
    # Train models
    # model = train("lgbm", train_x, train_y)

    # # Predict for testing data
    # predictions = model.predict(test_x)
    #
    # # Exponent on the predictions to fix for changing skewness
    # predictions_exp = np.exp(predictions)
    #
    # # Save submission
    # save_submission(model, predictions_exp, df_test)

    # Check linear regression coefficients
    model = train("lr", train_x, train_y)
    pairs = sorted([(train_x.columns.values.tolist()[i], model.coef_[i]) for i in range(len(model.coef_))], key=lambda x: x[1], reverse=True)
    print(pairs[:5])
    print(pairs[-5:])




if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=PearsonRConstantInputWarning)
    warnings.filterwarnings("ignore", category=PearsonRNearConstantInputWarning)
    main()