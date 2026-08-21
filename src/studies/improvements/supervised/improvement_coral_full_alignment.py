import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.supervised_baselines import SupervisedModels
from src.preprocessing.coral_alignment import alignment_of_coral_technique

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

def direction_of_testing(direction_name, x_train_raw, y_train, x_test_raw, y_test, output_dir):
    print(f"\nDirection of testing {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    coral_train_of_x = alignment_of_coral_technique(x_train_raw, x_test_raw)
    results = []
    for type_of_the_model in ["Random_Forest", "Gradient_Boosting", "Logistic_Regression", "Neural_Network"]:
        model_baseline = SupervisedModels(type_of_the_model=type_of_the_model)
        model_baseline.fit(x_train_raw, y_train.values)
        predictions_of_baseline = model_baseline.predict(x_test_raw)
        precision_of_baseline = precision_score(y_test, predictions_of_baseline, zero_division=0)
        recall_of_baseline = recall_score(y_test, predictions_of_baseline, zero_division=0)
        f1_score_of_baseline = f1_score(y_test, predictions_of_baseline, zero_division=0)
        model_coral = SupervisedModels(type_of_the_model=type_of_the_model)
        model_coral.fit(coral_train_of_x, y_train.values)
        predictions_of_coral = model_coral.predict(x_test_raw)
        precision_of_coral = precision_score(y_test, predictions_of_coral, zero_division=0)
        recall_of_coral = recall_score(y_test, predictions_of_coral, zero_division=0)
        f1_score_of_coral = f1_score(y_test, predictions_of_coral, zero_division=0)

        print(f"\n{type_of_the_model}")
        print(f" Baseline - Precision={precision_of_baseline:.4f}  Recall={recall_of_baseline:.4f}  F1={f1_score_of_baseline:.4f}")
        print(f" Coral - Precision={precision_of_coral:.4f}  Recall={recall_of_coral:.4f}  F1={f1_score_of_coral:.4f}")
        results.append({
            "Model": type_of_the_model,
            "Baseline_Precision": precision_of_baseline,
            "Baseline_Recall": recall_of_baseline,
            "Baseline_F1": f1_score_of_baseline,
            "Coral_Precision": precision_of_coral,
            "Coral_Recall": recall_of_coral,
            "Coral_F1": f1_score_of_coral,
            "F1_Improvement": f1_score_of_coral - f1_score_of_baseline
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "coral_vs_baseline.csv"), index=False)
    print(f"\n{results_df}")
    return results_df

def main():
    base_out = "results/studies/improvements/supervised/improvement_coral_full_alignment.py"
    total_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = direction_of_testing("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018,os.path.join(base_out, "2017_to_2018"))
    r["Direction"] = "2017_to_2018"
    total_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b,os.path.join(base_out, "unsw_to_2017"))
    r["Direction"] = "unsw_to_2017"
    total_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_testing("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b,os.path.join(base_out, "unsw_to_2018"))
    r["Direction"] = "unsw_to_2018"
    total_results.append(r)

    combined = pd.concat(total_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "coral_all_directions_summary.csv"), index=False)
    print(f"\nStudy of coral improvement has finished")
    print(combined[["Direction", "Model", "Baseline_F1", "Coral_F1", "F1_Improvement"]])

if __name__ == "__main__":
    main()