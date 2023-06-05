import pandas as pd
import numpy as np

from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import PowerTransformer,OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# outliers treatment
def winsorizer_left():
    outlier_left_columns = ['Humidity9am']
    winsorizer_left = Winsorizer(capping_method='quantiles', 
                                 tail='left', # cap left tail
                                 fold=0.05,
                                 variables=outlier_left_columns
                                 )
    return winsorizer_left

def winsorizer_right():
    outlier_right_columns = ['Rainfall','Evaporation','WindGustSpeed','WindSpeed9am','WindSpeed3pm']
    winsorizer_right = Winsorizer(capping_method='quantiles', 
                                  tail='right', # cap right tail
                                  fold=0.05,
                                  variables=outlier_right_columns
                                  )
    return winsorizer_right

def winsorizer_both():
    outlier_both_columns = ['Humidity3pm']
    winsorizer_both = Winsorizer(capping_method='quantiles', 
                                 tail='both', # cap both tails
                                 fold=0.05,
                                 variables=outlier_both_columns  
                                 )
    return winsorizer_both


# numerical transformation, categorical encoding and feature scaling
def get_data_transformer():

    numerical_columns = ['Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm',
                         'Humidity9am','Humidity3pm','Cloud9am','Cloud3pm','RainToday','AverageTemp']
    categorical_columns = ['Location','WindGustDir','WindDir9am','WindDir3pm','Pressure9am','Pressure3pm',
                           'ColourOfBoats']

    num_pipeline = Pipeline(
        steps=[
        ('yeo_johnson_transformer',PowerTransformer(method='yeo-johnson', standardize=False)),
        ('scaler',StandardScaler())
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
        ('one_hot_encoder',OneHotEncoder()),
        ('scaler',StandardScaler(with_mean=False))
        ]
    )

    preprocessor = ColumnTransformer(
        [
        ('num_pipeline',num_pipeline,numerical_columns),
        ('cat_pipeline',cat_pipeline,categorical_columns)
        ]
    )

    print('Data transformer created')

    return preprocessor


def initiate_data_transformation(df):
    
    # outliers treatment
    df = winsorizer_left().fit_transform(df)
    df = winsorizer_right().fit_transform(df)
    df = winsorizer_both().fit_transform(df)

    # drop columns
    col_to_drop = ['Date','RainTomorrow']

    X = df.drop(col_to_drop, axis=1)
    y = df['RainTomorrow']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    preprocessor = get_data_transformer()

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    print('Data transformation completed')

    return (X_train, X_test, y_train, y_test)