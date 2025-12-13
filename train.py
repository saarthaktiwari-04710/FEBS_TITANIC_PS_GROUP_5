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

df[['Deck','CabinNum','Side']]=df['Cabin'].str.split('/',expand=True)
df['GroupId']=df['PassengerId'].str.split('_').str[0]
df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
df['Deck'] = df.groupby('GroupId')['Deck'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x))
deck_mode = df['Deck'].mode()[0]
df['Deck'] = df['Deck'].fillna(deck_mode)
cabin_median = df['CabinNum'].median()
df['CabinNum'] = df['CabinNum'].fillna(cabin_median)
df['Side'] = df.groupby('GroupId')['Side'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x))
side_mode = df['Side'].mode()[0]
df['Side'] = df['Side'].fillna(side_mode)
df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')
