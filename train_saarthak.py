import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
train=pd.read_csv('train.csv')
df_train=pd.DataFrame(train)
si_mostfrequent=SimpleImputer(strategy='most_frequent')
df_train[['FoodCourt']]=si_mostfrequent.fit_transform(df_train[['FoodCourt']])
df_train[['RoomService']] = si_mostfrequent.fit_transform(df_train[['RoomService']])
df_train[['FoodCourt']] = si_mostfrequent.fit_transform(df_train[['FoodCourt']])
df_train[['CryoSleep']]=si_mostfrequent.fit_transform(df_train[['CryoSleep']])
si_missing=SimpleImputer(strategy='constant',fill_value='Missing')
df_train[['Name']]=si_missing.fit_transform(df_train[['Name']])

