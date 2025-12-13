import pandas as pd
import numpy as np
from preprocessing_TC import preprocess_TC
from preprocessing_saarthak import preprocess_saarthak
df = pd.read_csv("train.csv")

df = preprocess_TC(df)
df = preprocess_saarthak(df)
