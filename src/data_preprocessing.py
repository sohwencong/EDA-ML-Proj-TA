import pandas as pd
import numpy as np

def initiate_data_preprocessing(df):

    # drop rows of duplicated values in 'Date' and 'Location'
    df = df.drop_duplicates(subset=['Date','Location'], keep='first')
    # reset index
    df.reset_index(drop=True, inplace=True)

    # convert negative values in 'Sunshine' to positive
    df.loc[:,'Sunshine'] = abs(df['Sunshine'])

    # replace missing values of 'WindDir9am' that has 'WindSpeed9am' of 0 with 'NoWind'
    df.loc[:,'WindDir9am'] = np.where((df['WindDir9am'].isna()) & (df['WindSpeed9am']==0), 
                                'NoWind', df['WindDir9am'])
    # replace missing values of 'WindDir3pm' that has 'WindSpeed3pm' of 0 with 'NoWind'
    df.loc[:,'WindDir3pm'] = np.where((df['WindDir3pm'].isna()) & (df['WindSpeed3pm']==0), 
                                'NoWind', df['WindDir3pm'])
    
    # Convert strings to uppercase in 'Pressure9am'
    df.loc[:,'Pressure9am'] = df['Pressure9am'].str.upper()
    # Convert strings to uppercase in 'Pressure3pm'
    df.loc[:,'Pressure3pm'] = df['Pressure3pm'].str.upper()


    # replace missing values in 'RainToday'
    conditions = [(df['Rainfall'] > 1.0), (df['Rainfall'] <= 1.0)]
    cases = ['Yes', 'No']
    df.loc[:,'RainToday'] = np.select(conditions, cases)

    # drop all rows with NaN (missing) values and reset index
    df = df.dropna().reset_index(drop=True)

    # replace strings with integers for 'RainToday' and 'RainTomorrow'
    to_replace = {'Yes': 1,
                  'No': 0
                 }
    df = df.replace({'RainToday': to_replace,
                     'RainTomorrow': to_replace
                    }
                   )
    
    print('Data preprocessing completed')
    
    return df