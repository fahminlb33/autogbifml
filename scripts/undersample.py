import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load training data
df_train = pd.read_parquet("../dataset/merged/train.parquet")

# find minimum sample by region and target
min_sample = df_train[["country", "target"]].value_counts().min()

# sample from main dataset
df_undersampled = pd.concat([
    df_train[(df_train["country"] == "africa") & (df_train["target"] == 0)].sample(n=min_sample, random_state=42),
    df_train[(df_train["country"] == "africa") & (df_train["target"] == 1)].sample(n=min_sample, random_state=42),
    df_train[(df_train["country"] == "australia") & (df_train["target"] == 0)].sample(n=min_sample, random_state=42),
    df_train[(df_train["country"] == "australia") & (df_train["target"] == 1)].sample(n=min_sample, random_state=42),
])

# print stats
print("===== BEFORE UNDER SAMPLING")
print(df_train[["country", "target"]].value_counts().sort_index())

print("===== AFTER UNDER SAMPLING")
print(df_undersampled[["country", "target"]].value_counts().sort_index())

# save dataset
df_undersampled.to_parquet("../dataset/train-undersampled.parquet")
