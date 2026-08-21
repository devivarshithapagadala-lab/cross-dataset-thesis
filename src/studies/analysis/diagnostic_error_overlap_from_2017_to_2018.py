import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.preprocessing import preprocess
from src.models.autoencoder import AutoencoderForUnsupervisedModels

def error_overlap():
    print("Training on the dataset cicids2017 & testing on the dataset csccicids2018")
    training = pd.read_csv( "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv" )
    training = training.sample(n=50000,random_state=42).reset_index(drop=True)
    x_train, y_train, scaler = preprocess(training,scaler=None,tag_of_the_dataset="CIC")
    testing = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
    testing.columns = testing.columns.str.strip()
    benign = testing[testing["Label"].astype(str).str.strip().str.lower() == "benign" ].sample(n=45000,random_state=42)
    attack = testing[testing["Label"].astype(str).str.strip().str.lower() != "benign"].sample(n=5000,random_state=42)
    testing = pd.concat( [benign, attack] ).sample(frac=1,random_state=42).reset_index(drop=True)
    x_test, y_test = preprocess(testing,scaler=scaler,tag_of_the_dataset="CIC")
    model = AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1])
    model.fit(x_train,y_train.values,epochs=5)
    model.model.eval()
    with torch.no_grad():
        test_of_tensor = torch.tensor(x_test, dtype=torch.float32).to(model.device)
        reconstructions = model.model(test_of_tensor)
        errors = torch.mean((test_of_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
    y_test_arr = y_test.values
    errors_of_benign = errors[y_test_arr == 0]
    attack_errors = errors[y_test_arr == 1]
    print("\nsummary of raw error distribution")
    print(f"errors of benign > minimum - {errors_of_benign.min():.4f},"
          f"median - {np.median(errors_of_benign):.4f},"
          f"95th percentile - {np.percentile(errors_of_benign, 95):.4f},"
          f"maximum - {errors_of_benign.max():.4f}")
    print(f"errors of attack > minimum - {attack_errors.min():.4f},"
          f"median - {np.median(attack_errors):.4f},"
          f"95th percentile - {np.percentile(attack_errors, 95):.4f},"
          f"maximum - {attack_errors.max():.4f}")
    percentiles_to_test = [50, 60, 70, 75, 80, 85, 90, 95, 99]
    results = []
    for p in percentiles_to_test:
        threshold = np.percentile(errors_of_benign, p)
        predictions = np.where(errors > threshold, 1, 0)
        tp = np.sum((predictions == 1) & (y_test_arr == 1))
        fp = np.sum((predictions == 1) & (y_test_arr == 0))
        fn = np.sum((predictions == 0) & (y_test_arr == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        results.append({"Percentile": p,"Threshold_Value": threshold,"Precision": round(precision, 4),
            "Recall": round(recall, 4),"F1": round(f1, 4)
        })
    results_of_df = pd.DataFrame(results)
    print("\n percentile sweep of fine grain with only the calibration of benign")
    print(results_of_df)
    output_folder = "results/studies/analysis/diagnostic_error_overlap_from_2017_to_2018"
    os.makedirs(output_folder, exist_ok=True)
    results_of_df.to_csv(os.path.join(output_folder, "fine_grained_percentile_sweep.csv"),index=False)
    plt.figure(figsize=(9, 5))
    plt.hist(errors_of_benign, bins=80, alpha=0.6, label="BENIGN", color="steelblue")
    plt.hist(attack_errors, bins=80, alpha=0.6, label="ATTACK", color="firebrick")
    plt.title("Error overlap of reconstruction - from the dataset CICIDS2017 to CSE-CIC-IDS2018")
    plt.xlabel("MSE Mean Squared Error")
    plt.ylabel("Count of sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "error_overlap_histogram.png"), dpi=300)
    plt.close()
if __name__ == "__main__":
    error_overlap()