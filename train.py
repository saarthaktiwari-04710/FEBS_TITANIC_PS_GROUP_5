import pandas as pd
import numpy as np
from preprocessing_TC import preprocess_TC
import preprocessing_saarthak
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

#Transported Count Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
df['Transported'].value_counts().plot(kind='bar')

plt.xlabel("Transported")
plt.ylabel("Count")
plt.title("Transported Count Plot")
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()

#Age distribution graph
from scipy.stats import gaussian_kde
plt.figure(figsize=(8, 6))
plt.hist(age, bins=40, density=True)
kde = gaussian_kde(age)
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# HomePlanet vs Transported
plt.figure(figsize=(6, 4))
homeplanet_counts = pd.crosstab(df['HomePlanet'], df['Transported'])
homeplanet_counts.plot(kind='bar')
plt.xlabel("HomePlanet")
plt.ylabel("Count")
plt.title("HomePlanet vs Transported")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


