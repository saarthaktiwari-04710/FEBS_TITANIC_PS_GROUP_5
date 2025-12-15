import pandas as pd
import numpy as np
import pickle

df = pd.read_csv("test.csv")

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
