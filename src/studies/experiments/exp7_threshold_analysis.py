import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.autoencoder import AutoencoderForUnsupervisedModels

def load_pool_of_cic(path, benign_n, attack_n, seed=42):
    df = load_csv_after_removing_duplicates(path)
    benign = df[df["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = df[df["Label"].astype(str).str.strip().str.lower() != "benign"]
    a = attack.sample(n=min(attack_n, len(attack)), random_state=seed, replace=(len(attack) < attack_n))
    b = benign.sample(n=min(benign_n, len(benign)), random_state=seed, replace=(len(benign) < benign_n))
    return pd.concat([b, a]).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_source_of_unsw(path, n=50000, seed=42):
    df = load_csv_after_removing_duplicates(path)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
set_of_thresholds = ["mean_2std", "mean_3std", "mean_4std", "percentile95", "percentile99"]

def direction_of_testing(direction_name, x_train, y_train, x_test, y_test, output_dir):
    print(f"\nTesting the direction - {direction_name}\n")
    os.makedirs(output_dir, exist_ok=True)
    model = AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1])
    model.fit(x_train, y_train.values, epochs=5)
    results = []
    for threshold in set_of_thresholds:
        print(f"\nTesting of the threshold - {threshold}")
        metrics = model.evaluation_of_threshold(x_test, y_test.values, threshold)
        results.append({"Type_of_threshold": threshold, "Value_of_threshold": metrics["Threshold"],
                        "Accuracy": metrics["Accuracy"], "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"], "F1": metrics["F1"], "ROC_AUC": metrics["ROC_AUC"]})
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(output_dir, "threshold_summary.csv"), index=False)
    print(summary)
    return summary

def main():
    base_out = "results/studies/experiments/exp7_threshold_analysis"
    all_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = direction_of_testing("CICIDS2017 -> CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018,os.path.join(base_out, "2017_to_2018"))
    r["Direction"] = "2017_to_2018"
    all_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 -> CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b,os.path.join(base_out, "unsw_to_2017"))
    r["Direction"] = "unsw_to_2017"
    all_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 -> CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b,os.path.join(base_out, "unsw_to_2018"))
    r["Direction"] = "unsw_to_2018"
    all_results.append(r)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "threshold_all_directions_summary.csv"), index=False)
    print(combined[["Direction", "Type_of_threshold", "Value_of_threshold", "Recall", "F1", "ROC_AUC"]])

if __name__ == "__main__":
    main()