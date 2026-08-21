import os
import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from sklearn.ensemble import IsolationForest
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

contamination_options_of_isolation_forest = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NU_options_of_one_class_svm = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
contamination_of_local_outlier_factor = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
weight_options_of_ensemble = [(0.25, 0.25, 0.25, 0.25), (0.40, 0.20, 0.20, 0.20), (0.20, 0.40, 0.20, 0.20),
            (0.20, 0.20, 0.40, 0.20), (0.20, 0.20, 0.20, 0.40), (0.50, 0.50, 0.00, 0.00), (0.00, 0.50, 0.00, 0.50)]
percentiles_of_threshold = [50, 60, 70, 80, 85, 90, 95]
fraction_of_calibration = 0.05

def direction_of_testing(name_of_the_direction, source_of_x_train, y_train_source, x_target_full, y_target_full, output_dir):
    print(f"\n Final pipeline that is selected automatically {name_of_the_direction}\n")
    os.makedirs(output_dir, exist_ok=True)
    pool_calibration_of_x, evaluation_of_x, calib_pool_calibration_of_y, evaluation_of_y = train_test_split(x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    calibration_of_n = max(2, int(len(pool_calibration_of_x) * fraction_of_calibration))
    idx = np.random.RandomState(42).choice(len(pool_calibration_of_x), size=calibration_of_n, replace=False)
    calibration_of_x = pool_calibration_of_x[idx]
    calibration_of_y = calib_pool_calibration_of_y.values[idx]
    print(f"samples of calibration{calibration_of_n} & Held out evaluation - {evaluation_of_x.shape[0]}")
    benign_of_x_train = source_of_x_train[y_train_source.values == 0]
    candidates = []

    for c in contamination_options_of_isolation_forest:
        model = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({
            "name": f"Isolation_Forest(contamination={c})",
            "calib_f1": f1,
            "eval_metrics": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))
        })

    for nu in NU_options_of_one_class_svm:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({
            "name": f"One_Class_SVM(nu={nu})",
            "calib_f1": f1,
            "eval_metrics": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))
        })

    for c in contamination_of_local_outlier_factor:
        model = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True, n_jobs=-1)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({
            "name": f"Local_Outlier_Factor(contamination={c})",
            "calib_f1": f1,
            "eval_metrics": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                             f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))
        })

    autoencoder_model = AutoencoderForUnsupervisedModels(input_dim=source_of_x_train.shape[1])
    autoencoder_model.fit(source_of_x_train, y_train_source.values, epochs=5)
    threshold_options_of_autoencoder = ["percentile50", "percentile60", "percentile70", "percentile75",
                                        "percentile80", "percentile85", "percentile90",
                                        "mean_2std", "mean_3std", "mean_4std", "percentile95", "percentile99"]
    for t in threshold_options_of_autoencoder:
        metrics_of_calibration = autoencoder_model.evaluation_of_threshold(calibration_of_x, calibration_of_y, t)
        metrics_of_evaluations = autoencoder_model.evaluation_of_threshold(evaluation_of_x, evaluation_of_y.values, t)
        candidates.append({
            "name": f"Autoencoder(threshold={t})",
            "calib_f1": metrics_of_calibration["F1"],
            "eval_metrics": (metrics_of_evaluations["Precision"], metrics_of_evaluations["Recall"], metrics_of_evaluations["F1"])
        })

    isolation_forest_score_calibration = -IsolationForest(contamination=0.10, random_state=42, n_jobs=-1).fit(benign_of_x_train).score_samples(calibration_of_x)
    one_class_svm_score_calibration = -OneClassSVM(kernel="rbf", gamma="scale", nu=0.10).fit(benign_of_x_train).decision_function(calibration_of_x)
    local_outlier_factor_fitted = LocalOutlierFactor(n_neighbors=20, contamination=0.10, novelty=True, n_jobs=-1).fit(benign_of_x_train)
    local_outlier_factor_score_calibration = -local_outlier_factor_fitted.decision_function(calibration_of_x)
    autoencoder_model.model.eval()
    with torch.no_grad():
        tensor_calibration = torch.tensor(calibration_of_x, dtype=torch.float32).to(autoencoder_model.device)
        reconstructed_calibration = autoencoder_model.model(tensor_calibration)
        autoencoder_score_calibration = torch.mean((tensor_calibration - reconstructed_calibration) ** 2, dim=1).cpu().numpy()
        tensor_evaluation = torch.tensor(evaluation_of_x, dtype=torch.float32).to(autoencoder_model.device)
        reconstructed_evaluation = autoencoder_model.model(tensor_evaluation)
        autoencoder_score_evaluation = torch.mean((tensor_evaluation - reconstructed_evaluation) ** 2, dim=1).cpu().numpy()
    isolation_forest_score_evaluation = -IsolationForest(contamination=0.10, random_state=42, n_jobs=-1).fit(benign_of_x_train).score_samples(evaluation_of_x)
    one_class_svm_score_evaluation = -OneClassSVM(kernel="rbf", gamma="scale", nu=0.10).fit(benign_of_x_train).decision_function(evaluation_of_x)
    local_outlier_factor_score_evaluation = -local_outlier_factor_fitted.decision_function(evaluation_of_x)
    bounds = {
        "if": (isolation_forest_score_calibration.min(), isolation_forest_score_calibration.max()),
        "ocsvm": (one_class_svm_score_calibration.min(), one_class_svm_score_calibration.max()),
        "lof": (local_outlier_factor_score_calibration.min(), local_outlier_factor_score_calibration.max()),
        "ae": (autoencoder_score_calibration.min(), autoencoder_score_calibration.max()),
    }
    normalized_calibration_of_isolation_forest_c_n = scores_of_normalization(isolation_forest_score_calibration, *bounds["if"])
    normalized_calibration_of_one_class_svm = scores_of_normalization(one_class_svm_score_calibration, *bounds["ocsvm"])
    normalized_calibration_of_local_outlier_factor = scores_of_normalization(local_outlier_factor_score_calibration, *bounds["lof"])
    normalized_calibration_of_autoencoder = scores_of_normalization(autoencoder_score_calibration, *bounds["ae"])
    normalized_evaluation_of_isolation_forest = scores_of_normalization(isolation_forest_score_evaluation, *bounds["if"])
    normalized_evaluation_of_one_class_svm = scores_of_normalization(one_class_svm_score_evaluation, *bounds["ocsvm"])
    normalized_evaluation_of_local_outlier_factor = scores_of_normalization(local_outlier_factor_score_evaluation, *bounds["lof"])
    normalized_evaluation_of_autoencoder = scores_of_normalization(autoencoder_score_evaluation, *bounds["ae"])

    for weights in weight_options_of_ensemble:
        combined_c = weights[0]*normalized_calibration_of_isolation_forest_c_n + weights[1]*normalized_calibration_of_one_class_svm + weights[2]*normalized_calibration_of_local_outlier_factor + weights[3]*normalized_calibration_of_autoencoder
        combined_e = weights[0]*normalized_evaluation_of_isolation_forest + weights[1]*normalized_evaluation_of_one_class_svm + weights[2]*normalized_evaluation_of_local_outlier_factor + weights[3]*normalized_evaluation_of_autoencoder
        for pct in percentiles_of_threshold:
            threshold = np.percentile(combined_c, pct)
            predictions_of_calibration = np.where(combined_c > threshold, 1, 0)
            f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
            predictions_of_evaluations = np.where(combined_e > threshold, 1, 0)
            candidates.append({
                "name": f"Ensemble(weights={weights}, pct={pct})",
                "calib_f1": f1,
                "eval_metrics": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                 recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                 f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))
            })

    candidates_df = pd.DataFrame([
        {"Configuration": c["name"], "Calibration_F1": c["calib_f1"],
         "HeldOut_Precision": c["eval_metrics"][0], "HeldOut_Recall": c["eval_metrics"][1],
         "HeldOut_F1": c["eval_metrics"][2]}
        for c in candidates
    ])
    candidates_df.to_csv(os.path.join(output_dir, "all_candidates_ranked.csv"), index=False)
    success = candidates_df.loc[candidates_df["Calibration_F1"].idxmax()]
    print(f"\n selection made through calibration, {calibration_of_n} labeled samples - {success['Configuration']}")
    print(f"Held out result - Precision={success['HeldOut_Precision']:.4f}  "
          f"Recall={success['HeldOut_Recall']:.4f}  F1={success['HeldOut_F1']:.4f}")
    success_df = pd.DataFrame([success])
    success_df.to_csv(os.path.join(output_dir, "selected_winner.csv"), index=False)
    return success

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_unsupervised_auto_selected_pipeline"
    summary = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    w = direction_of_testing("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"))
    summary.append({"Direction": "2017_to_2018", **w.to_dict()})

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    w = direction_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"))
    summary.append({"Direction": "unsw_to_2017", **w.to_dict()})

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    w = direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"))
    summary.append({"Direction": "unsw_to_2018", **w.to_dict()})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_out, "final_pipeline_summary.csv"), index=False)
    print(summary_df[["Direction", "Configuration", "HeldOut_F1"]])

if __name__ == "__main__":
    main()