import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.preprocessing.preprocessing import preprocess

df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
df.columns = df.columns.str.strip()
benign = df[df["Label"].str.lower() == "benign"].sample( n=2500,random_state=42)
attack = df[df["Label"].str.lower() != "benign"].sample( n=2500,random_state=42)
df = pd.concat([benign, attack]).sample( frac=1,random_state=42)
X, y, scaler = preprocess(df)
PrincipalComponentAnalysis = PCA(n_components=2)
X_of_pca = PrincipalComponentAnalysis.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_of_pca[y == 0, 0], X_of_pca[y == 0, 1], s=10, alpha=0.5, label="Benign")
plt.scatter(X_of_pca[y == 1, 0], X_of_pca[y == 1, 1], s=10, alpha=0.5, label="Attack")
plt.title("Principal Component Analysis Projection - CICIDS2018")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
os.makedirs("results/studies/experiments/exp3_pca_visualization",exist_ok=True)
plt.savefig("results/studies/experiments/exp3_pca_visualization/pca_cic2018.png",dpi=300)
plt.show()