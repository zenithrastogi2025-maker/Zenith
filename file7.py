import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

print("Maximum Values:")
print(df.max())

print("\nMinimum Values:")
print(df.min())
