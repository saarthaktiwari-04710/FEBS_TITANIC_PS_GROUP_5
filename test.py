import pandas as pd
import numpy as np
import pickle
import preprocessing_saarthak

df = pd.read_csv("test.csv")
df[['RoomService']]=si_RS.transform(df[['RoomService']])
df[['FoodCourt']]=si_FC.transform(df[['FoodCourt']])
df[['Spa']]=si_SPA.transform(df[['Spa']])
df[['CryoSleep']]=si_CS.transform(df[['CryoSleep']])
df[['Name']]=si_Name.transform(df[['Name']])

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
