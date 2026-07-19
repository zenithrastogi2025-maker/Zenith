import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

sorted_df = df.sort_values(by="sepal length (cm)")

print(sorted_df.head(15))
