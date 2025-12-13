import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
train=pd.read_csv('train.csv')
df_train=pd.DataFrame(train)
si_mostfrequent=SimpleImputer(strategy='most_frequent')
df_train[['FoodCourt']]=si_mostfrequent.fit_transform(df_train[['FoodCourt']])
df_train[['RoomService']] = si_mostfrequent.fit_transform(df_train[['RoomService']])
df_train[['FoodCourt']] = si_mostfrequent.fit_transform(df_train[['FoodCourt']])
si_missing=SimpleImputer(strategy='constant',fill_value='Missing')
df_train[['Name']]=si_missing.fit_transform(df_train[['Name']])
si=SimpleImputer(strategy='most_frequent')
df_train[['CryoSleep']]=si.fit_transform(df_train[['CryoSleep']])
df_train[['Deck','Cabin_Num','Side']]=df_train['Cabin'].str.split('/',expand=True)
df_train['Group']=df_train['PassengerId'].str.split('_').str[0]
df_train['Cabin_Num'] = pd.to_numeric(df_train['Cabin_Num'], errors='coerce')
df_train['Deck'] = df_train.groupby('Group')['Deck'].transform(lambda x: x.fillna(x.mode()[0] if len(x.mode()) else x))
deck_mode = df_train['Deck'].mode()[0]
df_train['Deck'] = df_train['Deck'].fillna(deck_mode)
cabin_median = df_train['Cabin_Num'].median()
df_train['Cabin_Num'] = df_train['Cabin_Num'].fillna(cabin_median)
df_train['Side'] = df_train.groupby('Group')['Side'].transform(lambda x: x.fillna(x.mode()[0] if len(x.mode()) else x))
side_mode = df_train['Side'].mode()[0]
df_train['Side'] = df_train['Side'].fillna(side_mode)

