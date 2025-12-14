import numpy as np
import pandas as pd
def preprocess(df):
    
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])
    df["Destination"] = df["Destination"].fillna(df["Destination"].mode()[0])
    df["VIP"] = df["VIP"].fillna(False)
    df["ShoppingMall"] = df["ShoppingMall"].fillna(0)
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["VRDeck"] = df["VRDeck"].fillna(0)
    return df
