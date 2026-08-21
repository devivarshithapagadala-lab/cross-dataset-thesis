import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from sklearn.model_selection import train_test_split
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

def scores_of_normalization(scores, ref_minimum, ref_maximum):
    if ref_maximum - ref_minimum < 1e-9:
        return np.zeros_like(scores)
    return np.clip((scores - ref_minimum) / (ref_maximum - ref_minimum), 0, 1)

Fractions_of_calibration = [0.01, 0.02, 0.05, 0.10]
weight_options_of_ensemble = [(0.25, 0.25, 0.25, 0.25),(0.40, 0.20, 0.20, 0.20),(0.20, 0.40, 0.20, 0.20),
                              (0.20, 0.20, 0.40, 0.20),(0.20, 0.20, 0.20, 0.40),(0.50, 0.50, 0.00, 0.00),
                              (0.00, 0.50, 0.00, 0.50)]
Percentiles_of_threshold = [50, 60, 70, 80, 85, 90, 95]

def direction_of_testing(direction_name, x_train_source, y_train_source, x_target_full, y_target_full, output_dir, lof_contamination=0.10):
    print(f"\nImprovement of ensemble {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    pool_of_calibration_of_x, evaluation_of_x, pool_of_calibration_of_y, evaluation_of_y = train_test_split(x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    benign_of_x_train = x_train_source[y_train_source.values == 0]
    isolation_forest_model = IsolationForest(contamination=0.10, random_state=42, n_jobs=-1)
    isolation_forest_model.fit(benign_of_x_train)
    one_class_svm_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
    one_class_svm_model.fit(benign_of_x_train)
    local_outlier_factor_model = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination, novelty=True, n_jobs=-1)
    local_outlier_factor_model.fit(benign_of_x_train)
    autoencoder_model = AutoencoderForUnsupervisedModels(input_dim=x_train_source.shape[1])
    autoencoder_model.fit(x_train_source, y_train_source.values, epochs=5)

    def scores_that_are_raw(X):
        isolation_forest_score = -isolation_forest_model.score_samples(X)
        one_class_svm_score = -one_class_svm_model.decision_function(X)
        local_outlier_factor_score = -local_outlier_factor_model.decision_function(X)
        autoencoder_model.model.eval()
        with torch.no_grad():
            x_of_tensor = torch.tensor(X, dtype=torch.float32).to(autoencoder_model.device)
            reconstructions = autoencoder_model.model(x_of_tensor)
            autoencoder_score = torch.mean((x_of_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        return isolation_forest_score, one_class_svm_score, local_outlier_factor_score, autoencoder_score

    results = []
    for fraction in Fractions_of_calibration:
        calibration_of_n = max(2, int(len(pool_of_calibration_of_x) * fraction))
        idx = np.random.RandomState(42).choice(len(pool_of_calibration_of_x), size=calibration_of_n, replace=False)
        calibration_of_x = pool_of_calibration_of_x[idx]
        calibration_of_y = pool_of_calibration_of_y.values[idx]
        isolation_forest_calibration, one_class_svm_calibration, local_outlier_factor_calibration, autoencoder_calibration = scores_that_are_raw(calibration_of_x)
        norm_bounds = {
            "if": (isolation_forest_calibration.min(), isolation_forest_calibration.max()),
            "ocsvm": (one_class_svm_calibration.min(), one_class_svm_calibration.max()),
            "lof": (local_outlier_factor_calibration.min(), local_outlier_factor_calibration.max()),
            "ae": (autoencoder_calibration.min(), autoencoder_calibration.max()),
        }
        normalized_calibration_of_isolation_forest = scores_of_normalization(isolation_forest_calibration, *norm_bounds["if"])
        normalized_calibration_of_one_class_svm = scores_of_normalization(one_class_svm_calibration, *norm_bounds["ocsvm"])
        normalized_calibration_of_local_outlier_factor = scores_of_normalization(local_outlier_factor_calibration, *norm_bounds["lof"])
        normalized_calibration_of_autoencoder = scores_of_normalization(autoencoder_calibration, *norm_bounds["ae"])
        best_f1, best_weights, best_percentage = -1, None, None
        for weights in weight_options_of_ensemble:
            combined_c = (weights[0] * normalized_calibration_of_isolation_forest + weights[1] * normalized_calibration_of_one_class_svm +
                          weights[2] * normalized_calibration_of_local_outlier_factor + weights[3] * normalized_calibration_of_autoencoder)
            for pct in Percentiles_of_threshold:
                threshold = np.percentile(combined_c, pct)
                predictions = np.where(combined_c > threshold, 1, 0)
                f1 = f1_score(calibration_of_y, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_weights, best_percentage = f1, weights, pct
        isolation_forest_evaluation, one_class_svm_e, local_outlier_factor_e, autoencoder_e = scores_that_are_raw(evaluation_of_x)
        normalized_evaluation_of_isolation_forest = scores_of_normalization(isolation_forest_evaluation, *norm_bounds["if"])
        normalized_evaluation_of_one_class_svm = scores_of_normalization(one_class_svm_e, *norm_bounds["ocsvm"])
        normalized_evaluation_of_local_outlier_factor = scores_of_normalization(local_outlier_factor_e, *norm_bounds["lof"])
        normalized_evaluation_of_autoencoder = scores_of_normalization(autoencoder_e, *norm_bounds["ae"])
        combined_evaluation = (best_weights[0] * normalized_evaluation_of_isolation_forest + best_weights[1] * normalized_evaluation_of_one_class_svm +
                      best_weights[2] * normalized_evaluation_of_local_outlier_factor + best_weights[3] * normalized_evaluation_of_autoencoder)
        evaluation_of_threshold = np.percentile((best_weights[0] * normalized_calibration_of_isolation_forest + best_weights[1] *
                                                 normalized_calibration_of_one_class_svm + best_weights[2] *
                                                 normalized_calibration_of_local_outlier_factor + best_weights[3] * normalized_calibration_of_autoencoder), best_percentage)
        evaluation_of_predictions = np.where(combined_evaluation > evaluation_of_threshold, 1, 0)
        precision = precision_score(evaluation_of_y, evaluation_of_predictions, zero_division=0)
        recall = recall_score(evaluation_of_y, evaluation_of_predictions, zero_division=0)
        f1 = f1_score(evaluation_of_y, evaluation_of_predictions, zero_division=0)
        print(f"fraction={fraction:.2f} ({calibration_of_n} samples,  weights of IsolationForest,OneClassSVM,LocalOutlierFactor,AutoEncoder={best_weights}"
              f"pct={best_percentage}  HeldOut: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        results.append({
            "Fraction": fraction, "N_Calib": calibration_of_n,
            "Weight_IF": best_weights[0], "Weight_OCSVM": best_weights[1],
            "Weight_LOF": best_weights[2], "Weight_AE": best_weights[3],
            "Threshold_Percentile": best_percentage,
            "Precision": precision, "Recall": recall, "F1": f1
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "ensemble_results.csv"), index=False)
    best_of_all = results_df.loc[results_df["F1"].idxmax()]
    print(f"\nThe configuration of ensemble which is best for {direction_name}:")
    print(best_of_all)
    return results_df

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_unsupervised_ensemble"
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
    combined.to_csv(os.path.join(base_out, "ensemble_all_directions_summary.csv"), index=False)
    print(f"\nstudy of improvement of ensemble has finished\n")
    best_as_per_the_direction = combined.loc[combined.groupby("Direction")["F1"].idxmax()]
    print(best_as_per_the_direction[["Direction", "Fraction", "F1"]])

if __name__ == "__main__":
    main()