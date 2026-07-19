import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["Total Length"] = df["sepal length (cm)"] + df["petal length (cm)"]

print(df.head(10))
