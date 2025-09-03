import pandas as pd
data= pd.read_csv("data.csv")
print(data.head())
print(data.isnull().sum())
data_cleaned = data.dropna(subset=['disease', 'symptoms'])
data_cleaned = data_cleaned.reset_index(drop=True)
print(data_cleaned.shape)
