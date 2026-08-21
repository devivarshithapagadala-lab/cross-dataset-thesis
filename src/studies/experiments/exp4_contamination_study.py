import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

training_raw = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
training_raw = training_raw.sample(n=50000, random_state=42)
X_train, y_train, scaler = preprocess(training_raw, scaler=None)

testing_raw = load_csv_after_removing_duplicates("data/CSE-CIC-IDS2018/02-14-2018.csv")
benign = testing_raw[testing_raw["Label"].str.lower() == "benign"].sample(n=45000, random_state=42)
attack = testing_raw[testing_raw["Label"].str.lower() != "benign"].sample(n=5000, random_state=42)
test_df = pd.concat([benign, attack]).sample(frac=1, random_state=42)
X_test, y_test = preprocess(test_df, scaler=scaler)

values_of_contamination = [0.01, 0.05, 0.10, 0.20, 0.30]
results = []
for c in values_of_contamination:
    print(f"\nCurrently running contamination - {c}")
    model = IsolationForestModel(contamination=c)
    model.fit(X_train)
    raw_prediction = model.model.predict(X_test)
    prediction = [
        1 if p == -1 else 0
        for p in raw_prediction
    ]
    results.append({
        "Contamination": c,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0)
    })
results_df = pd.DataFrame(results)
print(results_df)
os.makedirs("results/studies/experiments/exp4_contamination_study", exist_ok=True)
results_df.to_csv("results/studies/experiments/exp4_contamination_study/contamination_results.csv", index=False)
plt.figure(figsize=(8, 5))
plt.plot(results_df["Contamination"], results_df["F1"], marker="o")
plt.title("Sensitivity of Contamination for the model 'Isolation Forest' ")
plt.xlabel("Contamination")
plt.ylabel("F1 Score")
plt.grid(True)
plt.savefig("results/studies/experiments/exp4_contamination_study/contamination_plot.png", dpi=300)
plt.show()