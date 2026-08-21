import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates, features_that_are_common
from src.models.autoencoder import AutoencoderForUnsupervisedModels


def load_pool_of_cic(path, benign_n, attack_n, seed=42):
    df = load_csv_after_removing_duplicates(path)
    benign = df[df["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = df[df["Label"].astype(str).str.strip().str.lower() != "benign"]
    b = benign.sample(n=min(benign_n, len(benign)), random_state=seed, replace=(len(benign) < benign_n))
    a = attack.sample(n=min(attack_n, len(attack)), random_state=seed, replace=(len(attack) < attack_n))
    return pd.concat([b, a]).sample(frac=1, random_state=seed).reset_index(drop=True)

FEATURES_OF_UNSW_FILLED_WITH_0 = [
    'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Flow IAT Std',
    'Flow IAT Min', 'Fwd PSH Flags', 'Min Packet Length',
    'Packet Length Std', 'SYN Flag Count', 'RST Flag Count',
    'ACK Flag Count', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]
INDICES_FILLED_WITH_0 = [features_that_are_common.index(f) for f in FEATURES_OF_UNSW_FILLED_WITH_0]
CONTAMINATION_OPTIONS_OF_ISOLATION_FOREST = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
NU_OPTIONS_OF_ONE_CLASS_SVM = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CONTAMINATION_OPTIONS_OF_LOCAL_OUTLIER_FACTOR = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
THRESHOLD_OPTIONS_OF_AUTOENCODER = ["percentile50", "percentile60", "percentile70", "percentile75",
                        "percentile80", "percentile85", "percentile90",
                        "mean_2std", "mean_3std", "percentile95"]
FRACTION_OF_CALIBRATION = 0.05

def direction_of_testing(direction_name, full_source_of_x_train, y_train_source, full_target_of_x, full_target_of_y, output_dir):
    print(f"\nTest of reducing feature - {direction_name}\n")
    os.makedirs(output_dir, exist_ok=True)
    reduced_train_of_x = np.delete(full_source_of_x_train, INDICES_FILLED_WITH_0, axis=1)
    reduced_target_of_x = np.delete(full_target_of_x, INDICES_FILLED_WITH_0, axis=1)
    print(f"original count of features {full_source_of_x_train.shape[1]} & Reduced - {reduced_train_of_x.shape[1]}")
    pool_of_x_calibration, evaluation_of_x, pool_of_y_calibration, evaluation_of_y = train_test_split(
        reduced_target_of_x, full_target_of_y, test_size=0.5, random_state=42, stratify=full_target_of_y
    )
    calibration_of_n = max(2, int(len(pool_of_x_calibration) * FRACTION_OF_CALIBRATION))
    idx = np.random.RandomState(42).choice(len(pool_of_x_calibration), size=calibration_of_n, replace=False)
    calibration_of_x = pool_of_x_calibration[idx]
    calibration_of_y = pool_of_y_calibration.values[idx]

    benign_of_x_train = reduced_train_of_x[y_train_source.values == 0]
    candidates = []

    for c in CONTAMINATION_OPTIONS_OF_ISOLATION_FOREST:
        model = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({"name": f"Isolation_Forest(c={c})", "calib_f1": f1,
                           "eval": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))})

    for nu in NU_OPTIONS_OF_ONE_CLASS_SVM:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({"name": f"One_Class_SVM(nu={nu})", "calib_f1": f1,
                           "eval": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))})

    for c in CONTAMINATION_OPTIONS_OF_LOCAL_OUTLIER_FACTOR:
        model = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True, n_jobs=-1)
        model.fit(benign_of_x_train)
        predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
        f1 = f1_score(calibration_of_y, predictions_of_calibration, zero_division=0)
        predictions_of_evaluations = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
        candidates.append({"name": f"Local_Outlier_Factor(c={c})", "calib_f1": f1,
                           "eval": (precision_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    recall_score(evaluation_of_y, predictions_of_evaluations, zero_division=0),
                                    f1_score(evaluation_of_y, predictions_of_evaluations, zero_division=0))})

    ae_model = AutoencoderForUnsupervisedModels(input_dim=reduced_train_of_x.shape[1])
    ae_model.fit(reduced_train_of_x, y_train_source.values, epochs=5)
    for t in THRESHOLD_OPTIONS_OF_AUTOENCODER:
        metrics_of_calibration = ae_model.evaluation_of_threshold(calibration_of_x, calibration_of_y, t)
        metrics_of_evaluation = ae_model.evaluation_of_threshold(evaluation_of_x, evaluation_of_y.values, t)
        candidates.append({"name": f"Autoencoder(t={t})", "calib_f1": metrics_of_calibration["F1"],
                           "eval": (metrics_of_evaluation["Precision"], metrics_of_evaluation["Recall"], metrics_of_evaluation["F1"])})

    candidates_df = pd.DataFrame([
        {"Configuration": c["name"], "Calibration_F1": c["calib_f1"],
         "HeldOut_Precision": c["eval"][0], "HeldOut_Recall": c["eval"][1], "HeldOut_F1": c["eval"][2]}
        for c in candidates
    ])
    candidates_df.to_csv(os.path.join(output_dir, "reduced_feature_candidates.csv"), index=False)

    result_of_best_row = candidates_df.loc[candidates_df["Calibration_F1"].idxmax()]
    print(f"\nreduced features - {result_of_best_row['Configuration']}")
    print(f"Held out - P={result_of_best_row['HeldOut_Precision']:.4f} R={result_of_best_row['HeldOut_Recall']:.4f} F1={result_of_best_row['HeldOut_F1']:.4f}")
    return result_of_best_row


def main():
    base_out = "results/studies/experiments/exp9_improvement_unsupervised_reduced_features_unsw"
    summary = []

    train_of_unsw = load_csv_after_removing_duplicates("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    sample_of_unsw_train = train_of_unsw.sample(n=min(50000, len(train_of_unsw)), random_state=42).reset_index(drop=True)
    unsw_train_of_x, unsw_train_of_y, scaler_of_unsw = preprocess(sample_of_unsw_train, scaler=None, tag_of_the_dataset="UNSW")

    test_of_2017 = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017, y_test_of_2017 = preprocess(test_of_2017, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    w = direction_of_testing("UNSW-NB15 towards CICIDS2017", unsw_train_of_x, unsw_train_of_y,
                             x_test_of_2017, y_test_of_2017, os.path.join(base_out, "unsw_to_2017"))
    summary.append({"Direction of testing": "unsw_to_2017", **w.to_dict()})

    test_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_2018, y_test_2018 = preprocess(test_2018, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    w = direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", unsw_train_of_x, unsw_train_of_y,
                             x_test_2018, y_test_2018, os.path.join(base_out, "unsw_to_2018"))
    summary.append({"Direction of testing": "unsw_to_2018", **w.to_dict()})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(base_out, "reduced_feature_summary.csv"), index=False)
    print(f"\nstudy of reducing features has finished\n")
    print(summary_df[["Direction of testing", "Configuration", "HeldOut_F1"]])

if __name__ == "__main__":
    main()