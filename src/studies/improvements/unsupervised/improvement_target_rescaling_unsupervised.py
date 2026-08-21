import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.one_class_svm import OneClassSVMModel
from sklearn.metrics import precision_score, recall_score, f1_score
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel
from src.models.local_outlier_factor import LocalOutlierFactorModel

def load_pool_of_cic(path, benign_n, attack_n, seed=42):
    df = load_csv_after_removing_duplicates(path)
    benign = df[df["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = df[df["Label"].astype(str).str.strip().str.lower() != "benign"]
    a = attack.sample(n=min(attack_n, len(attack)), random_state=seed, replace=(len(attack) < attack_n))
    b = benign.sample(n=min(benign_n, len(benign)), random_state=seed, replace=(len(benign) < benign_n))
    return pd.concat([b, a]).sample(frac=1, random_state=seed).reset_index(drop=True)

def preprocess_of_raw_features(df, dataset_tag):
    df = df.copy()
    x, y, scaler = preprocess(df, scaler=None, tag_of_the_dataset=dataset_tag)
    x_raw = scaler.inverse_transform(x)
    return x_raw, y

def predict_through_attribute_of_model(model, x_test):
    import numpy as np
    raw = model.model.predict(x_test)
    return np.where(raw == -1, 1, 0)

def direction_of_testing(direction_name, train_df, train_tag, test_df, test_tag, output_dir, lof_contamination=0.10):
    print(f"\nDirection of testing {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    baseline_of_x_train, y_train, source_scaler = preprocess(train_df, scaler=None, tag_of_the_dataset=train_tag)
    baseline_of_x_test, y_test = preprocess(test_df, scaler=source_scaler, tag_of_the_dataset=test_tag)
    raw_train_of_x, _ = preprocess_of_raw_features(train_df, train_tag)
    raw_test_of_x, _ = preprocess_of_raw_features(test_df, test_tag)
    scaler_target = StandardScaler()
    scaler_target.fit(raw_test_of_x)
    rescaled_train_of_x = scaler_target.transform(raw_train_of_x)
    rescaled_test_of_x = scaler_target.transform(raw_test_of_x)
    results = []
    configurations_of_model = {
        "Isolation_Forest": lambda: IsolationForestModel(contamination=0.10),
        "One_Class_SVM": lambda: OneClassSVMModel(nu=0.10),
        "Local_Outlier_Factor": lambda: LocalOutlierFactorModel(n_neighbors=20, contamination=lof_contamination),
    }
    for name, constructor in configurations_of_model.items():
        model_baseline = constructor()
        model_baseline.fit(baseline_of_x_train[y_train.values == 0])
        predictions_of_baseline = predict_through_attribute_of_model(model_baseline, baseline_of_x_test)
        precision_of_baseline = precision_score(y_test, predictions_of_baseline, zero_division=0)
        recall_of_baseline = recall_score(y_test, predictions_of_baseline, zero_division=0)
        f1_score_of_baseline = f1_score(y_test, predictions_of_baseline, zero_division=0)
        rescaled_model = constructor()
        rescaled_model.fit(rescaled_train_of_x[y_train.values == 0])
        predictions_of_rescaled = predict_through_attribute_of_model(rescaled_model, rescaled_test_of_x)
        precision_of_rescaled = precision_score(y_test, predictions_of_rescaled, zero_division=0)
        recall_of_rescaled = recall_score(y_test, predictions_of_rescaled, zero_division=0)
        f1_score_of_rescaled = f1_score(y_test, predictions_of_rescaled, zero_division=0)

        print(f"\n{name}")
        print(f" source rescaled baseline - Precision={precision_of_baseline:.4f}  Recall={recall_of_baseline:.4f}  F1={f1_score_of_baseline:.4f}")
        print(f"rescaled target - Precision={precision_of_rescaled:.4f}  Recall={recall_of_rescaled:.4f}  F1={f1_score_of_rescaled:.4f}")
        results.append({
            "Model": name,
            "Baseline_Precision": precision_of_baseline,
            "Baseline_Recall": recall_of_baseline,
            "Baseline_F1": f1_score_of_baseline,
            "Rescaled_Precision": precision_of_rescaled,
            "Rescaled_Recall": recall_of_rescaled,
            "Rescaled_F1": f1_score_of_rescaled,
            "F1_Improvement": f1_score_of_rescaled - f1_score_of_baseline
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "rescaling_vs_baseline.csv"), index=False)
    print(f"\n{results_df}")
    return results_df

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_target_rescaling_unsupervised"
    all_results = []
    directions = [
        ("CICIDS2017 towards CSE-CIC-IDS2018", "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", "CIC",
         "data/CSE-CIC-IDS2018/02-14-2018.csv", "CIC", 45000, 5000, "2017_to_2018", 0.10),
        ("UNSW-NB15 towards CICIDS2017", "data/UNSW_NB15/UNSW_NB15_training-set.csv", "UNSW",
         "data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", "CIC", 45000, 5000, "unsw_to_2017", 0.01),
        ("UNSW-NB15 towards CSE-CIC-IDS2018", "data/UNSW_NB15/UNSW_NB15_training-set.csv", "UNSW",
         "data/CSE-CIC-IDS2018/02-14-2018.csv", "CIC", 45000, 5000, "unsw_to_2018", 0.01),
    ]
    for name_of_direction, train_path, train_tag, test_path, test_tag, test_b, test_a, folder, lof_c in directions:
        training_raw = load_csv_after_removing_duplicates(train_path)
        train_df = training_raw.sample(n=min(50000, len(training_raw)), random_state=42).reset_index(drop=True)
        test_df = load_pool_of_cic(test_path, test_b, test_a)
        r = direction_of_testing(name_of_direction, train_df, train_tag, test_df, test_tag, os.path.join(base_out, folder), lof_contamination=lof_c)
        r["Direction"] = folder
        all_results.append(r)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "rescaling_all_directions_summary.csv"), index=False)
    print(f"\nStudy of the improvement of rescaling the target has finished\n")
    print(combined[["Direction", "Model", "Baseline_F1", "Rescaled_F1", "F1_Improvement"]])

if __name__ == "__main__":
    main()