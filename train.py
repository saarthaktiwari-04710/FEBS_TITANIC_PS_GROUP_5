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
age = df['Age'].dropna()
plt.figure(figsize=(8, 6))
plt.hist(age, bins=40, density=True)
kde = gaussian_kde(age)
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# Age vs Transported

plt.figure(figsize=(8, 6))
plt.hist(df[df['Transported'] == True]['Age'],
         bins=40,
         alpha=0.6,
         label='Transported')
plt.hist(df[df['Transported'] == False]['Age'],
         bins=40,
         alpha=0.6,
         label='Not Transported')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution by Transported")
plt.legend()
plt.tight_layout()
plt.show()

# HomePlanet vs Transported

plt.figure(figsize=(6, 4))
homeplanet_counts = pd.crosstab(df['HomePlanet'], df['Transported'])
homeplanet_counts.plot(kind='bar')
plt.xlabel("HomePlanet")
plt.ylabel("Count")
plt.title("HomePlanet vs Transported")
plt.tight_layout()
plt.show()

# Destination vs Transported

plt.figure(figsize=(6, 4))
destination_counts = pd.crosstab(df['Destination'], df['Transported'])
destination_counts.plot(kind='bar')
plt.xlabel("Destination")
plt.ylabel("Count")
plt.title("Destination vs Transported")
plt.tight_layout()
plt.show()

# Deck vs Transported

plt.figure(figsize=(6, 4))
deck_counts = pd.crosstab(df['Deck'], df['Transported'])
deck_counts.plot(kind='bar')
plt.xlabel("Deck")
plt.ylabel("Count")
plt.title("Deck vs Transported")
plt.tight_layout()
plt.show()

# Side vs Transported

plt.figure(figsize=(5, 4))
side_counts = pd.crosstab(df['Side'], df['Transported'])
side_counts.plot(kind='bar')
plt.xlabel("Side")
plt.ylabel("Count")
plt.title("Side vs Transported")
plt.tight_layout()
plt.show()

#Boxplot of transported vs total spending

spend_false = df[df["Transported"] == False]["TotalSpending"]
spend_true = df[df["Transported"] == True]["TotalSpending"]
plt.figure(figsize=(6, 4))
plt.boxplot([spend_false, spend_true], labels=["Not Transported", "Transported"])
plt.xlabel("Transported")
plt.ylabel("Total Spending")
plt.title("Total Spending vs Transported")
plt.show()

#Graph between cryosleep and transported

cryo_plot = (df.groupby("CryoSleep")["Transported"].apply(lambda x: x.mean()))
plt.figure(figsize=(5, 4))
cryo_plot.plot(kind="bar")
plt.xlabel("CryoSleep")
plt.ylabel("Proportion Transported")
plt.title("CryoSleep vs Transported")
plt.tight_layout()
plt.show()

# Dropping columns that are not useful 
df = df.drop(columns=['PassengerId','Name', 'Cabin'])

categorical_cols = ['HomePlanet', 'Destination','Deck', 'Side','VIP', 'CryoSleep']
# One-Hot Encoding
df = pd.get_dummies( df, columns=categorical_cols,  drop_first=True)

#  (Transported: True/False → 1/0)
df['Transported'] = df['Transported'].astype(int)




