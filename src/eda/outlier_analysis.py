import os
import pandas as pd

os.makedirs("results/eda/outliers", exist_ok=True )
df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
numeric = df.select_dtypes(include="number")
rows = []
for col in numeric.columns:
    q1 = numeric[col].quantile(0.25)
    q3 = numeric[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric[col] < q1 - 1.5*iqr) | (numeric[col] > q3 + 1.5*iqr)).sum()
    rows.append({ "Feature_as_columns": col, "Outliers": outliers })
pd.DataFrame(rows).to_csv("results/eda/outliers/outlier_summary.csv", index=False)