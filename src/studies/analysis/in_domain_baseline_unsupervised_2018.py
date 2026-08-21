import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import preprocess
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderForUnsupervisedModels
from src.models.one_class_svm import OneClassSVMModel
from src.models.local_outlier_factor import LocalOutlierFactorModel

def in_domain_baseline_for_unsupervised_models_2018():
    print("Training and testing of unsupervised models on same dataset CSE-CIC-IDS2018")
    raw = pd.read_csv("data/CSE-CIC-IDS2018/02-14-2018.csv")
    raw.columns = raw.columns.str.strip()
    benign = raw[raw["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = raw[raw["Label"].astype(str).str.strip().str.lower() != "benign"]
    samples_of_benign = benign.sample(n=min(45000, len(benign)), random_state=42)
    samples_of_attack = attack.sample(n=min(5000, len(attack)), random_state=42)
    full = pd.concat([samples_of_benign, samples_of_attack]).reset_index(drop=True)
    labels_to_start_with = full["Label"].astype(str).str.strip().str.lower().apply(lambda x: 0 if x == "benign" else 1)
    train_df, test_df = train_test_split(full,test_size=0.20,random_state=42,stratify=labels_to_start_with)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    x_train, y_train, scaler = preprocess(train_df, scaler=None, tag_of_the_dataset="CIC")
    x_test, y_test = preprocess(test_df, scaler=scaler, tag_of_the_dataset="CIC")
    print(f"\nshape of the processed train data- {x_train.shape}")
    print(f"shape of the processed test data-  {x_test.shape}")
    x_train_benign = x_train[y_train.values == 0]
    print(f"shape of the Benign subset of the training data- {x_train_benign.shape}")
    models = {
        "Isolation_Forest": IsolationForestModel(contamination=0.10),
        "Adaptive_Autoencoder": AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1]),
        "One_Class_SVM": OneClassSVMModel(nu=0.10),
        "Local_Outlier_Factor": LocalOutlierFactorModel(n_neighbors=20, contamination=0.10)
    }
    output_dir = "results/studies/analysis/in_domain_baseline_unsupervised_2018"
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        print(f"\nTraining of the model {name} on the same dataset CSE-CIC-IDS2018")
        if name == "Adaptive_Autoencoder":
            model.fit(x_train, y_train.values, epochs=5)
        else:
            model.fit(x_train_benign)
        print(f"Evaluation of the model {name} on the same dataset CSE-CIC-IDS2018")
        model.evaluate(x_test, y_test.values, output_dir=output_dir)
    print("\nBaseline of the in domain of unsupervised models are finished.")

if __name__ == "__main__":
    in_domain_baseline_for_unsupervised_models_2018()