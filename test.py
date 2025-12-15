import pandas as pd
import numpy as np
import pickle

import preprocessing_saarthak 
from preprocessing_TC import preprocess
df = pd.read_csv("test.csv")

df = preprocess(df)
df = preprocessing_saarthak.preprocess_saarthak(df)

df[['Deck','CabinNum','Side']]=df['Cabin'].str.split('/',expand=True)
df['GroupId']=df['PassengerId'].str.split('_').str[0]
df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
df['Deck'] = df.groupby('GroupId')['Deck'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
deck_mode = df['Deck'].mode()[0]
df['Deck'] = df['Deck'].fillna(deck_mode)
cabin_median = df['CabinNum'].median()
df['CabinNum'] = df['CabinNum'].fillna(cabin_median)
df['Side'] = df.groupby('GroupId')['Side'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
side_mode = df['Side'].mode()[0]
df['Side'] = df['Side'].fillna(side_mode)
df['GroupSize'] = df.groupby('GroupId')['PassengerId'].transform('count')

spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
df['TotalSpending'] = df[spending_cols].sum(axis=1)
passenger_ids = df['PassengerId']

# Dropping columns that are not useful 
df = df.drop(columns=['PassengerId','Name', 'Cabin','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'])

categorical_cols = ['HomePlanet', 'Destination','Deck', 'Side','VIP', 'CryoSleep']
# One-Hot Encoding
df = pd.get_dummies( df, columns=categorical_cols,  drop_first=True)

df['Transported'] = df['Transported'].astype(int)
X = df.drop('Transported', axis=1)

#loading values of w and b from train.py
with open('model_artifacts.pkl', 'rb') as f:artifacts = pickle.load(f)    
w = artifacts['w']
b = artifacts['b']
logit_offset = artifacts['logit_offset']
scaler = artifacts['scaler']
train_cols = artifacts['columns']

# Scaling
X[['Age', 'TotalSpending']] = scaler.transform(X[['Age', 'TotalSpending']])
X = X.values.astype(float)

#predicting transported
logits = (X @ w + b).flatten() - logit_offset #we use matrix multiplication between X and w
probs = sigmoid(logits)
predictions = (probs > 0.5)

#saving values in submission.csv
submission = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': predictions})
submission.to_csv('submission.csv', index=False)
