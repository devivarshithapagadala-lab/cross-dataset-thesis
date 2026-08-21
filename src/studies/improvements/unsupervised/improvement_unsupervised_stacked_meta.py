import os
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
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

def directions_of_testing(direction_name, source_of_x_train, y_train_source, full_target_of_x, full_target_of_y, output_dir, lof_contamination=0.10):
    print(f"\nClassifier of stacked meta - {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    pool_calibration_of_x, evaluation_of_x, pool_calibration_of_y, evaluation_of_y = train_test_split(
        full_target_of_x, full_target_of_y, test_size=0.5, random_state=42, stratify=full_target_of_y)
    benign_of_x_train = source_of_x_train[y_train_source.values == 0]
    isolation_forest_model = IsolationForest(contamination=0.10, random_state=42, n_jobs=-1)
    isolation_forest_model.fit(benign_of_x_train)
    one_class_svm_model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
    one_class_svm_model.fit(benign_of_x_train)
    local_outlier_factor_model = LocalOutlierFactor(n_neighbors=20, contamination=lof_contamination, novelty=True, n_jobs=-1)
    local_outlier_factor_model.fit(benign_of_x_train)
    autoencoder_model = AutoencoderForUnsupervisedModels(input_dim=source_of_x_train.shape[1])
    autoencoder_model.fit(source_of_x_train, y_train_source.values, epochs=5)

    def matrix_of_score(X, bounds=None):
        score_of_isolation_forest = -isolation_forest_model.score_samples(X)
        score_of_one_class_svm = -one_class_svm_model.decision_function(X)
        score_of_local_outlier_factor = -local_outlier_factor_model.decision_function(X)
        autoencoder_model.model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(autoencoder_model.device)
            r = autoencoder_model.model(t)
            autoencoder_s = torch.mean((t - r) ** 2, dim=1).cpu().numpy()
        if bounds is None:
            bounds = {
                "if": (score_of_isolation_forest.min(), score_of_isolation_forest.max()), "ocsvm": (score_of_one_class_svm.min(), score_of_one_class_svm.max()),
                "lof": (score_of_local_outlier_factor.min(), score_of_local_outlier_factor.max()), "ae": (autoencoder_s.min(), autoencoder_s.max())
            }
        matrix = np.column_stack([
            scores_of_normalization(score_of_isolation_forest, *bounds["if"]),
            scores_of_normalization(score_of_one_class_svm, *bounds["ocsvm"]),
            scores_of_normalization(score_of_local_outlier_factor, *bounds["lof"]),
            scores_of_normalization(autoencoder_s, *bounds["ae"]),
        ])
        return matrix, bounds
    results = []
    for fraction in Fractions_of_calibration:
        calibration_of_n = max(4, int(len(pool_calibration_of_x) * fraction))
        idx = np.random.RandomState(42).choice(len(pool_calibration_of_x), size=calibration_of_n, replace=False)
        calibration_of_x = pool_calibration_of_x[idx]
        calibration_of_y = pool_calibration_of_y.values[idx]
        if len(np.unique(calibration_of_y)) < 2:
            print(f"  fraction - {fraction:.2f} - has skipped because the calibration has only 1 class")
            continue
        scores_of_calibration, bounds = matrix_of_score(calibration_of_x)
        scores_of_evaluation, _ = matrix_of_score(evaluation_of_x, bounds=bounds)
        meta_classifier = LogisticRegression(class_weight="balanced", max_iter=500)
        meta_classifier.fit(scores_of_calibration, calibration_of_y)
        predictions_of_evaluations = meta_classifier.predict(scores_of_evaluation)
        precision = precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0)
        recall = recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0)
        f1 = f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0)
        learned_weights = meta_classifier.coef_[0]
        print(f"fraction - {fraction:.2f} ({calibration_of_n} samples"
              f"learned_weights of IsolationForest,OneClassSVM,LocalOutlierFactor,AutoEncoder) - {np.round(learned_weights, 3)}  "
              f"HeldOut: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        results.append({
            "Fraction": fraction, "N_Calib": calibration_of_n,
            "Weight_IF": learned_weights[0], "Weight_OCSVM": learned_weights[1],
            "Weight_LOF": learned_weights[2], "Weight_AE": learned_weights[3],
            "Precision": precision, "Recall": recall, "F1": f1
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "stacked_meta_classifier_results.csv"), index=False)
    if len(results_df) > 0:
        best = results_df.loc[results_df["F1"].idxmax()]
        print(f"\nThe configuration which is best for the direction {direction_name}:")
        print(best)
    return results_df

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_unsupervised_stacked_meta"
    total_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = directions_of_testing("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"), lof_contamination=0.10)
    r["Direction"] = "2017_to_2018"
    total_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = directions_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"), lof_contamination=0.01)
    r["Direction"] = "unsw_to_2017"
    total_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = directions_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"), lof_contamination=0.01)
    r["Direction"] = "unsw_to_2018"
    total_results.append(r)

    combined = pd.concat(total_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "stacked_meta_all_directions.csv"), index=False)
    print(f"\nStudy of stacked meta classifier has finished\n")
    best_per_direction = combined.loc[combined.groupby("Direction")["F1"].idxmax()]
    print(best_per_direction[["Direction", "Fraction", "F1"]])

if __name__ == "__main__":
    main()