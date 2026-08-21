import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.supervised_baselines import SupervisedModels
from sklearn.metrics import precision_score, recall_score, f1_score

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

Fractions_of_few_shot = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
Type_of_models = ["Random_Forest", "Gradient_Boosting"]

def fine_grained_label(direction_name, x_train_source, y_train_source, x_target_full, y_target_full, output_dir):
    print(f"\nFine grained few shot sweep - {direction_name}")
    os.makedirs(output_dir, exist_ok=True)
    few_shot_pool_of_x, evaluation_of_x, few_shot_pool_of_y, evaluation_of_y = train_test_split(
        x_target_full, y_target_full, test_size=0.5, random_state=42, stratify=y_target_full)
    results = []
    for model_type in Type_of_models:
        print(f"\n {model_type} ")
        for fraction in Fractions_of_few_shot:
            if fraction == 0.0:
                combined_train_of_x = x_train_source
                combined_train_of_y = y_train_source.values
                n_target_samples_used = 0
            else:
                samples_of_n = max(1, int(len(few_shot_pool_of_x) * fraction))
                idx = np.random.RandomState(42).choice(len(few_shot_pool_of_x), size=samples_of_n, replace=False)
                sample_target_of_x = few_shot_pool_of_x[idx]
                sample_target_of_y = few_shot_pool_of_y.values[idx]
                combined_train_of_x = np.vstack([x_train_source, sample_target_of_x])
                combined_train_of_y = np.concatenate([y_train_source.values, sample_target_of_y])
                n_target_samples_used = samples_of_n
            model = SupervisedModels(type_of_the_model=model_type)
            model.fit(combined_train_of_x, combined_train_of_y)
            predictions = model.predict(evaluation_of_x)
            precision = precision_score(evaluation_of_y, predictions, zero_division=0)
            recall = recall_score(evaluation_of_y, predictions, zero_division=0)
            f1 = f1_score(evaluation_of_y, predictions, zero_division=0)
            print(f"fraction={fraction:.3f} {n_target_samples_used} samples,  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
            results.append({
                "Model": model_type,
                "Target_Fraction": fraction,
                "Target_Samples_Used": n_target_samples_used,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "fine_grained_few_shot_sweep.csv"), index=False)
    print(f"\n{results_df.to_string(index=False)}")
    return results_df

def main():
    base_out = "results/studies/improvements/supervised/improvement_fine_grained_label_budget_study"
    all_results = []
    train_of_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
    train_of_2017 = train_of_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_of_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = fine_grained_label("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"))
    r["Direction"] = "2017_to_2018"
    all_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = fine_grained_label("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"))
    r["Direction"] = "unsw_to_2017"
    all_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = fine_grained_label("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"))
    r["Direction"] = "unsw_to_2018"
    all_results.append(r)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "fine_grained_all_directions_summary.csv"), index=False)
    print(f"\nFine grained label budget study finished\n")
    print(combined[["Direction", "Model", "Target_Fraction", "Target_Samples_Used", "F1"]])

if __name__ == "__main__":
    main()
