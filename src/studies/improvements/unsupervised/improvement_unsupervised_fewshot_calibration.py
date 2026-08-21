import os
import numpy as np
import pandas as pd
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

Fraction_of_calibration = [0.01, 0.02, 0.05, 0.10]
Contamination_options_of_isolation_forest = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NU_options_of_one_class_svm = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
Contamination_options_of_local_outlier_factor = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
Threshold_options_of_autoencoder = ["percentile50", "percentile60", "percentile70", "percentile75", "percentile80", "percentile85", "percentile90", "mean_2std", "mean_3std", "mean_4std", "percentile95", "percentile99"]

def configuration_evaluation(predictions, y_true):
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    return precision, recall, f1

def direction_of_testing(direction_name, x_train_source, y_train_source, x_target_full, y_target_full, output_dir):
    print(f"\nFew shot calibration with unsupervised methods - {direction_name}\n")
    os.makedirs(output_dir, exist_ok=True)
    pool_of_x_calibration, evaluation_of_x, pool_of_y_calibration, evaluation_of_Y = train_test_split(x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    print(f"Pool of calibration - {pool_of_x_calibration.shape[0]} & Held out evaluation - {evaluation_of_x.shape[0]}")
    benign_of_x_train = x_train_source[y_train_source.values == 0]
    results = []
    for fraction in Fraction_of_calibration:
        calibration_of_n = max(2, int(len(pool_of_x_calibration) * fraction))
        idx = np.random.RandomState(42).choice(len(pool_of_x_calibration), size=calibration_of_n, replace=False)
        calibration_of_x = pool_of_x_calibration[idx]
        calibration_of_y = pool_of_y_calibration.values[idx]
        print(f"\nfraction of calibration - {fraction:.2f} ({calibration_of_n} samples of labeled target)")
        best_f1, best_configuration, best_evaluation = -1, None, None
        for c in Contamination_options_of_isolation_forest:
            model = IsolationForest(contamination=c, random_state=42, n_jobs=-1)
            model.fit(benign_of_x_train)
            predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
            _, _, f1 = configuration_evaluation(predictions_of_calibration, calibration_of_y)
            if f1 > best_f1:
                predictions_of_evaluation = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
                best_f1, best_configuration, best_evaluation = f1, c, configuration_evaluation(predictions_of_evaluation, evaluation_of_Y)
        print(f"best contamination of isolation forest - {best_configuration}"
              f"Held out - P={best_evaluation[0]:.4f} R={best_evaluation[1]:.4f} F1={best_evaluation[2]:.4f}")
        results.append({"Model": "Isolation_Forest", "Fraction": fraction, "N_Calib": calibration_of_n, "Best_Config": best_configuration, "Precision": best_evaluation[0], "Recall": best_evaluation[1], "F1": best_evaluation[2]})

        best_f1, best_configuration, best_evaluation = -1, None, None
        for nu in NU_options_of_one_class_svm:
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
            model.fit(benign_of_x_train)
            predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
            _, _, f1 = configuration_evaluation(predictions_of_calibration, calibration_of_y)
            if f1 > best_f1:
                predictions_of_evaluation = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
                best_f1, best_configuration, best_evaluation = f1, nu, configuration_evaluation(predictions_of_evaluation, evaluation_of_Y)
        print(f"  One_Class_SVM     best_nu={best_configuration}  "
              f"Held out - P={best_evaluation[0]:.4f} R={best_evaluation[1]:.4f} F1={best_evaluation[2]:.4f}")
        results.append({"Model": "One_Class_SVM", "Fraction": fraction, "N_Calib": calibration_of_n,
                        "Best_Config": best_configuration, "Precision": best_evaluation[0],
                        "Recall": best_evaluation[1], "F1": best_evaluation[2]})

        best_f1, best_configuration, best_evaluation = -1, None, None
        for c in Contamination_options_of_local_outlier_factor:
            model = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True, n_jobs=-1)
            model.fit(benign_of_x_train)
            predictions_of_calibration = np.where(model.predict(calibration_of_x) == -1, 1, 0)
            _, _, f1 = configuration_evaluation(predictions_of_calibration, calibration_of_y)
            if f1 > best_f1:
                predictions_of_evaluation = np.where(model.predict(evaluation_of_x) == -1, 1, 0)
                best_f1, best_configuration, best_evaluation = f1, c, configuration_evaluation(predictions_of_evaluation, evaluation_of_Y)
        print(f"  Local_Outlier_Factor  best_contamination={best_configuration}  "
              f"HeldOut: P={best_evaluation[0]:.4f} R={best_evaluation[1]:.4f} F1={best_evaluation[2]:.4f}")
        results.append({"Model": "Local_Outlier_Factor", "Fraction": fraction, "N_Calib": calibration_of_n,
                        "Best_Config": best_configuration, "Precision": best_evaluation[0],
                        "Recall": best_evaluation[1], "F1": best_evaluation[2]})

        ae_model = AutoencoderForUnsupervisedModels(input_dim=x_train_source.shape[1])
        ae_model.fit(x_train_source, y_train_source.values, epochs=5)
        best_f1, best_configuration, best_evaluation = -1, None, None
        for type_of_threshold in Threshold_options_of_autoencoder:
            metrics_of_calibration = ae_model.evaluation_of_threshold(calibration_of_x, calibration_of_y, type_of_threshold)
            if metrics_of_calibration["F1"] > best_f1:
                metrics_of_evaluation = ae_model.evaluation_of_threshold(evaluation_of_x, evaluation_of_Y.values, type_of_threshold)
                best_f1 = metrics_of_calibration["F1"]
                best_configuration = type_of_threshold
                best_evaluation = (metrics_of_evaluation["Precision"], metrics_of_evaluation["Recall"], metrics_of_evaluation["F1"])
        print(f" best threshold of autoencoder - {best_configuration}  "
              f"Held out - P={best_evaluation[0]:.4f} R={best_evaluation[1]:.4f} F1={best_evaluation[2]:.4f}")
        results.append({"Model": "Autoencoder", "Fraction": fraction, "N_Calib": calibration_of_n,
                        "Best_Config": best_configuration, "Precision": best_evaluation[0],
                        "Recall": best_evaluation[1], "F1": best_evaluation[2]})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "unsupervised_fewshot_calibration.csv"), index=False)
    best_as_per_the_model = results_df.loc[results_df.groupby("Model")["F1"].idxmax()]
    print(f"\nconfiguration which is best per the model of the direction {direction_name}:")
    print(best_as_per_the_model.to_string(index=False))
    return results_df

def main():
    base_out = "results/studies/improvements/unsupervised/improvement_unsupervised_fewshot_calibration"
    total_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = direction_of_testing("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"))
    r["Direction"] = "2017_to_2018"
    total_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"))
    r["Direction"] = "unsw_to_2017"
    total_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"))
    r["Direction"] = "unsw_to_2018"
    total_results.append(r)

    combined = pd.concat(total_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "unsupervised_fewshot_all_directions.csv"), index=False)
    print(f"\nstudy of few shot calibration for unsupervised methods have finished\n")
    best_of_all = combined.loc[combined.groupby(["Direction", "Model"])["F1"].idxmax()]
    print(best_of_all[["Direction", "Model", "Fraction", "Best_Config", "F1"]])

if __name__ == "__main__":
    main()