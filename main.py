
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, boxcox_normmax, probplot
from scipy.special import boxcox1p
from sklearn.preprocessing import StandardScaler
from models import train, test, cv, get_model


def handle_missing_data(df):
    print("Handling missing data...")
    # Dealing with missing data
    # Remove all the variables that have missing data, except for Electrical we delete
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print("Removing features because of missing data:")
    print(", ".join(missing_data[missing_data['Total'] > 1].index))
    removed_features = (missing_data[missing_data['Total'] > 1]).index

    df = df.drop(removed_features, 1)
    df = df.drop(df.loc[df['Electrical'].isnull()].index)
    return df, removed_features

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

    # Define all the features that have absolute skewness above 0.5 as skewed and apply transformation
    high_skew = features_skew[features_skew.abs() > skew_thresh]

    # Apply Box-Cox transformation to skewed features
    for i in high_skew.index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

    return df

def remove_outliars(df):
    df = df.drop(df[df['Id'] == 1299].index)
    df = df.drop(df[df['Id'] == 524].index)
    return df

def clean_data(df):
    df, removed_features = handle_missing_data(df)
    df = remove_outliars(df)

    # Remove some of the features - should we? TODO
    # df = df.drop(['Street' ], axis=1)

    return df, removed_features

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

def preprocess_df_train(df_train):
    df_train, removed_features = clean_data(df_train)
    df_train = fix_skewness(df_train)
    df_train = feature_engineering(df_train)

    # Convert categorical data into dummy variables - one hot encoding
    df_train = pd.get_dummies(df_train)

    # Remove ID - helps nothing as it uniquely identify every entry
    df_train = df_train.drop(['Id'], axis=1)

    train_x = df_train.drop(['SalePrice'], axis=1)
    train_y = df_train["SalePrice"]
    return train_x, train_y, removed_features

def preprocess_df_test(df_test, removed_features):
    df_test = df_test.drop(removed_features, 1)
    df_test = fix_skewness(df_test)
    df_test = feature_engineering(df_test)

    # Convert categorical data into dummy variables - one hot encoding
    df_test = pd.get_dummies(df_test)

    # Remove ID - helps nothing as it uniquely identify every entry
    df_test = df_test.drop(['Id'], axis=1)

    return df_test

def main():
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    train_x, train_y, removed_features = preprocess_df_train(df_train)
    test_x = preprocess_df_test(df_test, removed_features)

    # Train models
    # cv(get_model("xgb"), train_x, train_y)
    xgb = train("lgbm", train_x, train_y)

    test(xgb, test_x, df_test)


if __name__ == '__main__':
    main()