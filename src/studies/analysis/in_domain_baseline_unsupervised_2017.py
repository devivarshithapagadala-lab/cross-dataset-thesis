import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import preprocess
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderForUnsupervisedModels
from src.models.one_class_svm import OneClassSVMModel
from src.models.local_outlier_factor import LocalOutlierFactorModel

def in_domain_baseline_for_unsupervised_models_2017():
    print("Training and testing on unsupervised model on same dataset 'CICIDS2017'")
    raw = pd.read_csv("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
    raw.columns = raw.columns.str.strip()
    benign = raw[raw["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = raw[raw["Label"].astype(str).str.strip().str.lower() != "benign"]
    benign_samples_of_benign = benign.sample(n=min(45000, len(benign)), random_state=42)
    attack_samples_of_attack = attack.sample(n=min(5000, len(attack)), random_state=42)
    full = pd.concat([benign_samples_of_benign, attack_samples_of_attack]).reset_index(drop=True)
    labels_to_start_with = full["Label"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x == "benign" else 1)
    train_df, test_df = train_test_split(full,test_size=0.20,random_state=42,stratify=labels_to_start_with)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    x_train, y_train, scaler = preprocess(train_df, scaler=None, tag_of_the_dataset="CIC")
    x_test, y_test = preprocess(test_df, scaler=scaler, tag_of_the_dataset="CIC")
    print(f"\nshape of the processed train data- {x_train.shape}")
    print(f"shape of the processed test data:  {x_test.shape}")
    x_train_benign = x_train[y_train.values == 0]
    print(f"shape of the benign subset of training- {x_train_benign.shape}")
    models = {
        "Isolation_Forest": IsolationForestModel(contamination=0.10),
        "Adaptive_Autoencoder": AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1]),
        "One_Class_SVM": OneClassSVMModel(nu=0.10),
        "Local_Outlier_Factor": LocalOutlierFactorModel(n_neighbors=20, contamination=0.10)
    }
    output_dir = "results/studies/analysis/in_domain_baseline_unsupervised_2017"
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        print(f"\n Training the model {name} on the same dataset CICIDS2017")
        if name == "Adaptive_Autoencoder":
            model.fit(x_train, y_train.values, epochs=5)
        else:
            model.fit(x_train_benign)
        print(f"Evaluation of the model {name} on the same dataset CICIDS2017 for testing")
        model.evaluate(x_test, y_test.values, output_dir=output_dir)
    print("\n Baseline of in domain unsupervised models have finished")

if __name__ == "__main__":
    in_domain_baseline_for_unsupervised_models_2017()