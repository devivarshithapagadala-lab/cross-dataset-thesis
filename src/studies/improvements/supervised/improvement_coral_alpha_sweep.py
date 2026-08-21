import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.supervised_baselines import SupervisedModels
from src.preprocessing.coral_alignment import alignment_of_coral_technique
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

Values_of_alpha = [0.0, 0.25, 0.5, 0.75, 1.0]
Type_of_models = ["Random_Forest", "Gradient_Boosting", "Logistic_Regression", "Neural_Network"]

def direction_of_sweep(direction_name, x_train, y_train, x_test, y_test, output_dir):
    print(f"\n Direction of the sweep {direction_name}\n")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for model_type in Type_of_models:
        print(f"\n {model_type} ")
        for alpha in Values_of_alpha:
            blended_train_of_x = alignment_of_coral_technique(x_train, x_test, alpha=alpha)
            model = SupervisedModels(type_of_the_model=model_type)
            model.fit(blended_train_of_x, y_train.values)
            predictions = model.predict(x_test)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            print(f"  alpha value = {alpha:.2f}  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
            results.append({
                "Model": model_type,
                "Alpha": alpha,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "coral_alpha_sweep.csv"), index=False)
    best_per_model = results_df.loc[results_df.groupby("Model")["F1"].idxmax()]
    print(f"\nThe best alpha per a model for the direction {direction_name}:")
    print(best_per_model[["Model", "Alpha", "Precision", "Recall", "F1"]])
    best_per_model.to_csv(os.path.join(output_dir, "coral_best_alpha_summary.csv"), index=False)
    return results_df

def main():
    base_out = "results/studies/improvements/supervised/improvement_coral_alpha_sweep"
    total_results = []
    train_dataset_2017 = load_csv_after_removing_duplicates("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
    train_dataset_2017 = train_dataset_2017.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train_of_2017, y_train_of_2017, scaler_of_2017 = preprocess(train_dataset_2017, scaler=None, tag_of_the_dataset="CIC")
    test_of_2018 = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018, y_test_of_2018 = preprocess(test_of_2018, scaler=scaler_of_2017, tag_of_the_dataset="CIC")
    r = direction_of_sweep("CICIDS2017 towards CSE-CIC-IDS2018", x_train_of_2017, y_train_of_2017, x_test_of_2018, y_test_of_2018, os.path.join(base_out, "2017_to_2018"))
    r["Direction"] = "2017_to_2018"
    total_results.append(r)

    train_of_unsw = load_source_of_unsw("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    x_train_of_unsw, y_train_of_unsw, scaler_of_unsw = preprocess(train_of_unsw, scaler=None, tag_of_the_dataset="UNSW")
    test_of_2017_b = load_pool_of_cic("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv", 45000, 5000)
    x_test_of_2017_b, y_test_of_2017_b = preprocess(test_of_2017_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_sweep("UNSW-NB15 towards CICIDS2017", x_train_of_unsw, y_train_of_unsw, x_test_of_2017_b, y_test_of_2017_b, os.path.join(base_out, "unsw_to_2017"))
    r["Direction"] = "unsw_to_2017"
    total_results.append(r)

    test_of_2018_b = load_pool_of_cic("data/CSE-CIC-IDS2018/02-14-2018.csv", 45000, 5000)
    x_test_of_2018_b, y_test_of_2018_b = preprocess(test_of_2018_b, scaler=scaler_of_unsw, tag_of_the_dataset="CIC")
    r = direction_of_sweep("UNSW-NB15 towards CSE-CIC-IDS2018", x_train_of_unsw, y_train_of_unsw, x_test_of_2018_b, y_test_of_2018_b, os.path.join(base_out, "unsw_to_2018"))
    r["Direction"] = "unsw_to_2018"
    total_results.append(r)

    combined = pd.concat(total_results, ignore_index=True)
    combined.to_csv(os.path.join(base_out, "coral_alpha_sweep_all_directions.csv"), index=False)
    print(f"\nStudy of coral alpha has finished\n")

if __name__ == "__main__":
    main()