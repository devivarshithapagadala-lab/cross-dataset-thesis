import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.preprocessing.preprocessing import preprocess

df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
df = df.sample( n=5000, random_state=42 )
X, y, scaler = preprocess(df)
corr = pd.DataFrame(X).corr()
plt.figure(figsize=(12,10))
sns.heatmap( corr, cmap="coolwarm", center=0 )
plt.title("visualization of Feature Correlation Heatmap")
plt.tight_layout()
os.makedirs("results/eda/correlation_analysis", exist_ok=True )
plt.savefig("results/eda/correlation_analysis/correlation_heatmap.png", dpi=300 )
plt.show()