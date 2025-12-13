import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
def preprocess_saarthak(df):
    si_mostfrequent=SimpleImputer(strategy='most_frequent')
    si_missing=SimpleImputer(strategy='constant',fill_value='Missing'
    df[['FoodCourt']] = si_mostfrequent.fit_transform(df[['FoodCourt']])
    df[['RoomService']] = si_mostfrequent.fit_transform(df[['RoomService']])
    df[['CryoSleep']] = si_mostfrequent.fit_transform(df[['CryoSleep']])
    df[['Name']]=si_missing.fit_transform(df[['Name']])
return df
