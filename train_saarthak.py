import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
si_RS = SimpleImputer(strategy='constant', fill_value=0)
si_FC = SimpleImputer(strategy='constant', fill_value=0)
si_CS=SimpleImputer(strategy='most_frequent')
si_Name=SimpleImputer(strategy='constant',fill_value='Missing')
def preprocess_saarthak(df):
    df[['FoodCourt']] = si_FC.fit_transform(df[['FoodCourt']])
    df[['RoomService']] = si_RS.fit_transform(df[['RoomService']])
    df[['CryoSleep']] = si_CS.fit_transform(df[['CryoSleep']])
    df[['Name']]=si_Name.fit_transform(df[['Name']])
return df
