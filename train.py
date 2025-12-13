import pandas as pd
import numpy as np
from preprocessing_TC import preprocess_TC
from preprocessing_saarthak import preprocess_saarthak
df = pd.read_csv("train.csv")

df = preprocess_TC(df)
df = preprocess_saarthak(df)

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)

#Handling misiing values in column cabin

df[['Deck','Cabin_Num','Side']]=df['Cabin'].str.split('/',expand=True)
df['Group']=df['PassengerId'].str.split('_').str[0]
df['Cabin_Num'] = pd.to_numeric(df['Cabin_Num'], errors='coerce')
df['Deck'] = df.groupby('Group')['Deck'].transform(lambda x: x.fillna(x.mode()[0] if len(x.mode()) else x))
deck_mode = df['Deck'].mode()[0]
df['Deck'] = df['Deck'].fillna(deck_mode)
cabin_median = df['Cabin_Num'].median()
df['Cabin_Num'] = df['Cabin_Num'].fillna(cabin_median)
df['Side'] = df.groupby('Group')['Side'].transform(lambda x: x.fillna(x.mode()[0] if len(x.mode()) else x))
side_mode = df['Side'].mode()[0]
df['Side'] = df['Side'].fillna(side_mode)
