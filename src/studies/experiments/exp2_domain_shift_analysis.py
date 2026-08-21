import os
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates

os.makedirs("results/studies/experiments/exp2_domain_shift_analysis", exist_ok=True)

df17 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv").sample(10000, random_state=42)
X17, y17, scaler = preprocess(df17, scaler=None, tag_of_the_dataset="CIC")
df18 = load_csv_after_removing_duplicates("data/CSE-CIC-IDS2018/02-14-2018.csv").sample(10000, random_state=42)
X18, y18 = preprocess(df18, scaler=scaler, tag_of_the_dataset="CIC")
df_unsw = load_csv_after_removing_duplicates("data/UNSW_NB15/UNSW_NB15_training-set.csv").sample(10000, random_state=42)
X_unsw, y_unsw = preprocess(df_unsw, scaler=scaler, tag_of_the_dataset="UNSW")

distance_from_2017_to_2018 = euclidean(X17.mean(axis=0), X18.mean(axis=0))
distance_from_unsw_to_2017 = euclidean(X_unsw.mean(axis=0), X17.mean(axis=0))
distance_from_unsw_to_2018 = euclidean(X_unsw.mean(axis=0), X18.mean(axis=0))

results = [{
    "Distance_2017_2018": distance_from_2017_to_2018,
    "Distance_UNSW_2017": distance_from_unsw_to_2017,
    "Distance_UNSW_2018": distance_from_unsw_to_2018
}]
pd.DataFrame(results).to_csv("results/studies/experiments/exp2_domain_shift_analysis/domain_shift_metrics.csv",index=False)
print("\nResults of shifting domain")
print("Euclidean Distance from 2017 to 2018 - ", distance_from_2017_to_2018)
print("Euclidean Distance from unsw to 2017 - ", distance_from_unsw_to_2017)
print("Euclidean Distance from unsw to 2018 - ", distance_from_unsw_to_2018)

results_of_wasserstein = []
for i in range(X17.shape[1]):
    wd = wasserstein_distance(X_unsw[:, i], X17[:, i])
    results_of_wasserstein.append({"Feature": i, "Distance_of_wasserstein_from_unsw_to_2017": wd})
    print(f"Feature {i} - Wasserstein between the datasets unsw vs 2017 = {wd:.4f}")

pd.DataFrame(results_of_wasserstein).to_csv("results/studies/experiments/exp2_domain_shift_analysis/wasserstein_distances.csv",index=False)