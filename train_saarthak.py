import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
def preprocess_saarthak(df):
si_mostfrequent=SimpleImputer(strategy='most_frequent')
    df[['FoodCourt']] = si_mostfrequent.fit_transform(df[['FoodCourt']])
    df[['RoomService']] = si_mostfrequent.fit_transform(df[['RoomService']])
    df[['CryoSleep']] = si_mostfrequent.fit_transform(df[['CryoSleep']])
    df['Name'] = df['Name'].fillna("Missing")
return df
