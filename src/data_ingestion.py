import sqlite3
import pandas as pd

import data_preprocessing
import data_transformation
import model_trainer

if __name__ == '__main__':

    con = sqlite3.connect('data/fishing.db')

    df = pd.read_sql_query('SELECT * FROM fishing', con)

    df = data_preprocessing.initiate_data_preprocessing(df)

    X_train, X_test, y_train, y_test = data_transformation.initiate_data_transformation(df)

    model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)