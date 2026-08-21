import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
df.columns = df.columns.str.strip()
counts = df["Label"].value_counts()
plt.figure(figsize=(8,5))
counts.plot(kind="bar")
plt.title("visualization of Class Distribution for the dataset cicids2018")
plt.ylabel("Examples")
plt.tight_layout()
os.makedirs("results/eda/class_distribution", exist_ok=True )
plt.savefig("results/eda/class_distribution/class_distribution.png", dpi=300 )
plt.show()