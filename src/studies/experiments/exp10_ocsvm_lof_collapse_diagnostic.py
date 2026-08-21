import os
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates


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

def direction_of_testing(direction_name, source_of_x_train, y_train_source, x_test, y_test, output_dir):
    print(f"\ndiagnosis of the collapse of lof and oneclass_svm {direction_name}\n")
    os.makedirs(output_dir, exist_ok=True)
    benign_of_x_train = source_of_x_train[y_train_source.values == 0]
    one_class_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.10)
    one_class_svm.fit(benign_of_x_train)
    one_class_svm_train_scores = one_class_svm.decision_function(benign_of_x_train)
    one_class_svm_test_scores = one_class_svm.decision_function(x_test)
    print(f"train scores of benign of one class svm, min={one_class_svm_train_scores.min():.4f}, "
          f"max={one_class_svm_train_scores.max():.4f}, mean={one_class_svm_train_scores.mean():.4f}")
    print(f"test scores of target of one class svm,  min={one_class_svm_test_scores.min():.4f}, "
          f"max={one_class_svm_test_scores.max():.4f}, mean={one_class_svm_test_scores.mean():.4f}")
    negative_test_percentage = (one_class_svm_test_scores < 0).mean() * 100
    print(f"samples of test percentage that scored as outlier, score < 0 - {negative_test_percentage:.2f}%")

    local_out_factor = LocalOutlierFactor(n_neighbors=20, contamination=0.10, novelty=True, n_jobs=-1)
    local_out_factor.fit(benign_of_x_train)
    local_outlier_factor_train_scores = local_out_factor.decision_function(benign_of_x_train)
    local_outlier_factor_test_scores = local_out_factor.decision_function(x_test)
    print(f"\ntrain scores of benign of lof, min={local_outlier_factor_train_scores.min():.4f}, "
          f"max={local_outlier_factor_train_scores.max():.4f}, mean={local_outlier_factor_train_scores.mean():.4f}")
    print(f"test scores of target of lof, min={local_outlier_factor_test_scores.min():.4f}, "
          f"max={local_outlier_factor_test_scores.max():.4f}, mean={local_outlier_factor_test_scores.mean():.4f}")
    negative_test_percentage_lof = (local_outlier_factor_test_scores < 0).mean() * 100
    print(f"samples of test percentage that scored as outlier, score < 0 - {negative_test_percentage_lof:.2f}%")

    minimal_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    results = []
    for val in minimal_values:
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=val)
        model.fit(benign_of_x_train)
        predictions = np.where(model.predict(x_test) == -1, 1, 0)
        percentage_of_predicted_attack = predictions.mean() * 100
        results.append({"Model": "One_Class_SVM", "Param": val, "Percentage_Predicted_Attack": percentage_of_predicted_attack})
        print(f"one class svm nu={val:.3f}: {percentage_of_predicted_attack:.2f}% of the test samples that are predicted as attack")

    for val in minimal_values:
        model = LocalOutlierFactor(n_neighbors=20, contamination=val, novelty=True, n_jobs=-1)
        model.fit(benign_of_x_train)
        predictions = np.where(model.predict(x_test) == -1, 1, 0)
        percentage_of_predicted_attack = predictions.mean() * 100
        results.append({"Model": "Local_Outlier_Factor", "Param": val, "Percentage_Predicted_Attack": percentage_of_predicted_attack})
        print(f"local outlier factor contamination={val:.3f}: {percentage_of_predicted_attack:.2f}% of the test samples that are predicted as attack")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "small_param_sweep.csv"), index=False)

    shift_of_feature = np.abs(benign_of_x_train.mean(axis=0) - x_test.mean(axis=0))
    top_shifted_idx = np.argsort(shift_of_feature)[::-1][:5]
    print(f"feature indices on top five by largest shift of mean  {top_shifted_idx.tolist()}")
    print(f"shift magnitudes of them {shift_of_feature[top_shifted_idx].tolist()}")
    shift_df = pd.DataFrame({"Feature_Index": range(len(shift_of_feature)), "Mean_Shift": shift_of_feature})
    shift_df.to_csv(os.path.join(output_dir, "feature_mean_shift.csv"), index=False)
    return results_df

def main():
    base_out = "results/studies/experiments/exp10_ocsvm_lof_collapse_diagnostic"
    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017 = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017, y_test_of_2017 = preprocess(test_of_2017, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    direction_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017, y_test_of_2017,os.path.join(base_out, "unsw_to_2017"))
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018, y_test_of_2018,os.path.join(base_out, "unsw_to_2018"))

if __name__ == "__main__":
    main()