import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target

print("First 10 Records:")
print(df.head(10))

print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)
