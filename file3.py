import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

print("Missing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())
