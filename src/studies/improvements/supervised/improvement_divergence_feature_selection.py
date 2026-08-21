import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as distance_of_wasserstein
from sklearn.model_selection import train_test_split
from src.models.supervised_baselines import SupervisedModels
from sklearn.metrics import precision_score, recall_score, f1_score
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates, features_that_are_common

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

def computation_of_feature_divergence(x_source, x_reference):
    n_features = x_source.shape[1]
    distances = []
    for i in range(n_features):
        d = distance_of_wasserstein(x_source[:, i], x_reference[:, i])
        distances.append(d)
    return np.array(distances)

Top_options_of_k = [0, 3, 5, 8, 10, 15]
Type_of_models = ["Random_Forest", "Gradient_Boosting", "Logistic_Regression", "Neural_Network"]

def direction_of_testing(direction_name, x_train, y_train, x_target_full, y_target_full, output_dir):
    print(f"\n Direction of testing {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    x_calib, x_eval, y_calib, y_eval = train_test_split(x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    print(f"Set of Calibration {x_calib.shape[0]} samples & Set of Evaluation held out on {x_eval.shape[0]} samples")
    divergence = computation_of_feature_divergence(x_train, x_calib)
    indices_of_ranked_features = np.argsort(divergence)[::-1]
    divergence_df = pd.DataFrame({"Feature": [features_that_are_common[i] for i in range(len(features_that_are_common))],
        "Wasserstein_Distance": divergence}).sort_values("Wasserstein_Distance", ascending=False)
    divergence_df.to_csv(os.path.join(output_dir, "feature_divergence_ranking.csv"), index=False)
    print("\n most divergent features which are on top five and is computed only on the data of calibration")
    print(divergence_df.head(5).to_string(index=False))

    results_of_calibration = []
    end_results = []
    for model_type in Type_of_models:
        print(f"\n {model_type} ")
        best_k = None
        best_calibration_f1 = -1.0
        for k in Top_options_of_k:
            remove_idx = indices_of_ranked_features[:k] if k > 0 else []
            reduced_train_of_x = np.delete(x_train, remove_idx, axis=1) if k > 0 else x_train
            reduced_calibration_of_x = np.delete(x_calib, remove_idx, axis=1) if k > 0 else x_calib
            model = SupervisedModels(type_of_the_model=model_type)
            model.fit(reduced_train_of_x, y_train.values)
            predictions_of_calibration = model.predict(reduced_calibration_of_x)
            f1_score_of_calibration = f1_score(y_calib, predictions_of_calibration, zero_division=0)
            results_of_calibration.append({"Model": model_type, "Features_Removed": k, "Calibration_F1": f1_score_of_calibration})
            print(f"  Calibration removed top {k}  F1={f1_score_of_calibration:.4f}")
            if f1_score_of_calibration > best_calibration_f1:
                best_calibration_f1 = f1_score_of_calibration
                best_k = k
        remove_idx = indices_of_ranked_features[:best_k] if best_k > 0 else []
        final_train_of_x = np.delete(x_train, remove_idx, axis=1) if best_k > 0 else x_train
        final_evaluation_of_x = np.delete(x_eval, remove_idx, axis=1) if best_k > 0 else x_eval
        final_model = SupervisedModels(type_of_the_model=model_type)
        final_model.fit(final_train_of_x, y_train.values)
        predictions_of_evaluation = final_model.predict(final_evaluation_of_x)
        precision_of_evaluation = precision_score(y_eval, predictions_of_evaluation, zero_division=0)
        recall_of_evaluation = recall_score(y_eval, predictions_of_evaluation, zero_division=0)
        f1_score_of_evaluation = f1_score(y_eval, predictions_of_evaluation, zero_division=0)
        print(f"Selected k - {best_k} which is considered through calibration"
              f"held out evaluation - Precision={precision_of_evaluation:.4f}  Recall={recall_of_evaluation:.4f}  F1={f1_score_of_evaluation:.4f}")
        end_results.append({
            "Model": model_type,
            "Selected_K": best_k,
            "Calibration_F1_at_Selection": best_calibration_f1,
            "Held_Out_Precision": precision_of_evaluation,
            "Held_Out_Recall": recall_of_evaluation,
            "Held_Out_F1": f1_score_of_evaluation
        })
    pd.DataFrame(results_of_calibration).to_csv(os.path.join(output_dir, "calibration_sweep_full.csv"), index=False)
    final_df = pd.DataFrame(end_results)
    final_df.to_csv(os.path.join(output_dir, "final_held_out_results.csv"), index=False)
    print(f"\nFinal results that are held out for the direction {direction_name}:")
    print(final_df.to_string(index=False))
    return final_df

def main():
    base_out = "results/studies/improvements/supervised/improvement_divergence_feature_selection"
    total_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
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
    combined.to_csv(os.path.join(base_out, "divergence_all_directions_final_summary.csv"), index=False)
    print(f"\nStudy of the selection of features guided by divergence has finished\n")
    print(combined[["Direction", "Model", "Selected_K", "Held_Out_F1"]])

if __name__ == "__main__":
    main()