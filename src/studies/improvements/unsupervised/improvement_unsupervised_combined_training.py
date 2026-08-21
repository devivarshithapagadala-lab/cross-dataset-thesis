import os
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from sklearn.metrics import precision_score, recall_score, f1_score
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
Fractions_of_benign_target = [0.0, 0.01, 0.02, 0.05, 0.10]

def direction_of_testing(name_of_direction, source_of_x_train, y_train_source, x_target_full, y_target_full, output_dir, lof_contamination=0.10):
    print(f"\nTraining of combined domain - {name_of_direction}")
    os.makedirs(output_dir, exist_ok=True)
    pool_of_x_target, evaluation_of_x, pool_of_y_target, evaluation_of_y = train_test_split(x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    pool_of_target_benign = pool_of_x_target[pool_of_y_target.values == 0]
    print(f"Availability of target benign pool for training - {len(pool_of_target_benign)} samples")
    print(f" Held out evaluation by combination of benign plus attack - {evaluation_of_x.shape[0]} samples")
    source_benign_of_x_train = source_of_x_train[y_train_source.values == 0]
    results = []
    for fraction in Fractions_of_benign_target:
        n_target_benign = int(len(pool_of_target_benign) * fraction)
        if n_target_benign > 0:
            idx = np.random.RandomState(42).choice(len(pool_of_target_benign), size=n_target_benign, replace=False)
            benign_of_combined_train = np.vstack([source_benign_of_x_train, pool_of_target_benign[idx]])
        else:
            benign_of_combined_train = source_benign_of_x_train
        print(f"\nFraction of benign target {fraction:.2f} ({n_target_benign} samples are added)")

        isolation_forest_model = IsolationForest(contamination=0.10, random_state=42, n_jobs=-1)
        isolation_forest_model.fit(benign_of_combined_train)
        predictions = np.where(isolation_forest_model.predict(evaluation_of_x) == -1, 1, 0)
        p, r, f1 = (precision_score(evaluation_of_y, predictions, zero_division=0), recall_score(evaluation_of_y, predictions, zero_division=0), f1_score(evaluation_of_y, predictions, zero_division=0))
        print(f"  Isolation Forest      P={p:.4f} R={r:.4f} F1={f1:.4f}")
        results.append({"Model": "Isolation_Forest", "Fraction": fraction, "N_Target_Benign": n_target_benign, "Precision": p, "Recall": r, "F1": f1})

        one_class_svm_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
        one_class_svm_model.fit(benign_of_combined_train)
        predictions = np.where(one_class_svm_model.predict(evaluation_of_x) == -1, 1, 0)
        p, r, f1 = (precision_score(evaluation_of_y, predictions, zero_division=0), recall_score(evaluation_of_y, predictions, zero_division=0), f1_score(evaluation_of_y, predictions, zero_division=0))
        print(f"  One Class SVM P={p:.4f} R={r:.4f} F1={f1:.4f}")
        results.append({"Model": "One_Class_SVM", "Fraction": fraction, "N_Target_Benign": n_target_benign, "Precision": p, "Recall": r, "F1": f1})

        local_outlier_factor_model = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination, novelty=True, n_jobs=-1)
        local_outlier_factor_model.fit(benign_of_combined_train)
        predictions = np.where(local_outlier_factor_model.predict(evaluation_of_x) == -1, 1, 0)
        p, r, f1 = (precision_score(evaluation_of_y, predictions, zero_division=0), recall_score(evaluation_of_y, predictions, zero_division=0), f1_score(evaluation_of_y, predictions, zero_division=0))
        print(f"  Local Outlier Factor  P={p:.4f} R={r:.4f} F1={f1:.4f}")
        results.append({"Model": "Local_Outlier_Factor", "Fraction": fraction, "N_Target_Benign": n_target_benign, "Precision": p, "Recall": r, "F1": f1})

        labels_of_autoencoder = np.zeros(len(benign_of_combined_train))
        autoencoder_model = AutoencoderForUnsupervisedModels(input_dim=benign_of_combined_train.shape[1])
        autoencoder_model.fit(benign_of_combined_train, labels_of_autoencoder, epochs=5)
        metrics = autoencoder_model.evaluation_of_threshold(evaluation_of_x, evaluation_of_y.values, "percentile90")
        print(f"  Autoencoder P = {metrics['Precision']:.4f} R={metrics['Recall']:.4f} F1={metrics['F1']:.4f}")
        results.append({"Model": "Autoencoder", "Fraction": fraction, "N_Target_Benign": n_target_benign, "Precision": metrics["Precision"], "Recall": metrics["Recall"], "F1": metrics["F1"]})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "combined_training_results.csv"), index=False)
    best_of_the_model = results_df.loc[results_df.groupby("Model")["F1"].idxmax()]
    print(f"\nconfiguration which is best per model for the direction {name_of_direction}:")
    print(best_of_the_model.to_string(index=False))
    return results_df

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_unsupervised_combined_training"
    total_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = direction_of_testing("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"), lof_contamination=0.10)
    r["Direction"] = "2017_to_2018"
    total_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"), lof_contamination=0.01)
    r["Direction"] = "unsw_to_2017"
    total_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"), lof_contamination=0.01)
    r["Direction"] = "unsw_to_2018"
    total_results.append(r)

    combined = pd.concat(total_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "combined_training_all_directions.csv"), index=False)
    print(f"\nStudy of training of combined domain has finished\n")
    best_of_all = combined.loc[combined.groupby(["Direction", "Model"])["F1"].idxmax()]
    print(best_of_all[["Direction", "Model", "Fraction", "F1"]])

if __name__ == "__main__":
    main()