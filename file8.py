import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

df = pd.DataFrame(digits.data)

df["Digit"] = digits.target

print(df["Digit"].value_counts())
