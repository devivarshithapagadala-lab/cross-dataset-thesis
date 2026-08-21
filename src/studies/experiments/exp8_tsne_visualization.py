import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.preprocessing.preprocessing import preprocess

df = pd.read_csv("data/CSE-CIC-IDS2018/02-14-2018.csv")
df = df.sample(n=3000,random_state=42)
X, y, scaler = preprocess(df)
t_Distributed_Stochastic_Neighbor_Embedding = TSNE(n_components=2, random_state=42)
X_of_tsne = t_Distributed_Stochastic_Neighbor_Embedding.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_of_tsne[y == 0, 0], X_of_tsne[y == 0, 1], s=10, alpha=0.5, label="Benign")
plt.scatter(X_of_tsne[y == 1, 0], X_of_tsne[y == 1, 1], s=10, alpha=0.5, label="Attack")
plt.title("Projection of t-Distributed Stochastic Neighbor Embedding - CICIDS2018")
plt.legend()
os.makedirs("results/studies/experiments/exp8_tsne_visualization/tsne",exist_ok=True)
plt.savefig("results/studies/experiments/exp8_tsne_visualization/tsne/tsne_cic2018.png",dpi=300)
plt.show()