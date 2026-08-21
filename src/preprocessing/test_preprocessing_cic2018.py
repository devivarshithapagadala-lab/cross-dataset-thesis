import pandas as pd
from src.preprocessing.preprocessing import preprocess

df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
X, y, _ = preprocess(df)
print("Shape of x-", X.shape)
print("Shape of y-", y.shape)
print("\nCount of the label")
print(y.value_counts())