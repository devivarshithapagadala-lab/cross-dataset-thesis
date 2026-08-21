import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from src.preprocessing.preprocessing import preprocess, features_that_are_common, load_csv_after_removing_duplicates

training_raw = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
training_raw = training_raw.sample(n=50000, random_state=42).reset_index(drop=True)
X_train, y_train, scaler = preprocess(training_raw, scaler=None, tag_of_the_dataset="CIC")

testing_raw = load_csv_after_removing_duplicates("data/CSE-CIC-IDS2018/02-14-2018.csv")
benign = testing_raw[testing_raw["Label"].astype(str).str.strip().str.lower() == "benign"].sample(n=45000, random_state=42)
attack = testing_raw[testing_raw["Label"].astype(str).str.strip().str.lower() != "benign"].sample(n=5000, random_state=42)
test_df = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
X_test, y_test = preprocess(test_df, scaler=scaler, tag_of_the_dataset="CIC")

groups_of_features = {
    "BASELINE": [],
    "Traffic_Volume": [
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packets/s",
        "Bwd Packets/s"
    ],
    "Packet_Length": [
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Packet Length Mean",
        "Packet Length Std"
    ],
    "Timing": [
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min"
    ],
    "Flags": [
        "SYN Flag Count",
        "RST Flag Count",
        "ACK Flag Count"
    ]
}
results = []
for name_of_the_group, removal_of_features in groups_of_features.items():
    print(f"\n Testing the removal of group of the feature - {name_of_the_group}")
    if len(removal_of_features) == 0:
        X_train_mod = X_train
        X_test_mod = X_test
    else:
        remove_idx = [
            features_that_are_common.index(col)
            for col in removal_of_features
            if col in features_that_are_common
        ]
        X_train_mod = np.delete(X_train, remove_idx, axis=1)
        X_test_mod = np.delete(X_test, remove_idx, axis=1)
    model = IsolationForest(contamination=0.10, random_state=42, n_jobs=-1)
    model.fit(X_train_mod)
    prediction = model.predict(X_test_mod)
    prediction = np.where(prediction == -1, 1, 0)
    f1 = f1_score(y_test, prediction, zero_division=0)
    precision = precision_score(y_test, prediction, zero_division=0)
    recall = recall_score(y_test, prediction, zero_division=0)
    results.append({
        "Feature_Group_Removed": name_of_the_group,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4)
    })
results_df = pd.DataFrame(results)
print("\n")
print("Results of feature ablation")
print(results_df)
os.makedirs("results/studies/experiments/exp5_feature_ablation", exist_ok=True)
results_df.to_csv("results/studies/experiments/exp5_feature_ablation/feature_ablation.csv", index=False)
plt.figure(figsize=(8, 5))
plt.bar(results_df["Feature_Group_Removed"], results_df["F1"])
plt.title("Study of feature ablation")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("results/studies/experiments/exp5_feature_ablation/feature_ablation_plot.png", dpi=300)
plt.close()