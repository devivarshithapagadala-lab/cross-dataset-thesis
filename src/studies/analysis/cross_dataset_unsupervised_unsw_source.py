import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderForUnsupervisedModels
from src.models.one_class_svm import OneClassSVMModel
from src.models.local_outlier_factor import LocalOutlierFactorModel

def load_pool_of_cic(path, benign_n, attack_n, seed=42):
    df = load_csv_after_removing_duplicates(path)
    benign = df[df["Label"].astype(str).str.strip().str.lower() == "benign"]
    attack = df[df["Label"].astype(str).str.strip().str.lower() != "benign"]
    a = attack.sample(n=min(attack_n, len(attack)), random_state=seed, replace=(len(attack) < attack_n))
    b = benign.sample(n=min(benign_n, len(benign)), random_state=seed, replace=(len(benign) < benign_n))
    return pd.concat([b, a]).sample(frac=1, random_state=seed).reset_index(drop=True)


def unsupervised_pipeline_with_unsw_as_source():
    training_raw = load_csv_after_removing_duplicates("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    training_sampled = training_raw.sample(n=min(50000, len(training_raw)), random_state=42).reset_index(drop=True)
    x_train, y_train, trained_scaler = preprocess(training_sampled, scaler=None, tag_of_the_dataset="UNSW")
    print(f" Normal Baseline Array, Shape of training set- {x_train.shape}")

    target_dataset_for_testing = {
        "CICIDS2017": {
            "path": "data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
            "tag": "CIC",
            "benign_samples": 45000,
            "attack_samples": 5000
        },
        "CICIDS2018": {
            "path": "data/CSE-CIC-IDS2018/02-14-2018.csv",
            "tag": "CIC",
            "benign_samples": 45000,
            "attack_samples": 5000
        }
    }

    models = {
        "Isolation_Forest": IsolationForestModel(contamination=0.10),
        "Adaptive_Autoencoder": AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1]),
        "One_Class_SVM": OneClassSVMModel(nu=0.10),
        "Local_Outlier_Factor": LocalOutlierFactorModel(n_neighbors=20, contamination=0.01)
    }
    for name, model in models.items():
        if name == "Adaptive_Autoencoder":
            model.fit(x_train, y_train.values, epochs=5)
        else:
            model.fit(x_train)
    print("Training of the four unsupervised models is finished.")

    for name_of_target, config in target_dataset_for_testing.items():
        print(f"\nEvaluating the shift of Domain of unsw towards {name_of_target.upper()}")
        try:
            testing_raw = load_csv_after_removing_duplicates(str(config["path"]))
            pool_of_benign = testing_raw[testing_raw['Label'].astype(str).str.strip().str.lower() == 'benign']
            pool_of_attack = testing_raw[testing_raw['Label'].astype(str).str.strip().str.lower() != 'benign']
            sample_of_b = pool_of_benign.sample(n=config["benign_samples"], random_state=42, replace=True)
            sample_of_a = pool_of_attack.sample(n=config["attack_samples"], random_state=42, replace=True)
            testing_sampled = pd.concat([sample_of_b, sample_of_a]).sample(frac=1, random_state=42).reset_index(drop=True)
            x_test, y_test = preprocess(testing_sampled, scaler=trained_scaler, tag_of_the_dataset=config["tag"])
            target_output_directory = os.path.join("results", "studies", "analysis", "cross_dataset_unsupervised_unsw_source")
            os.makedirs(target_output_directory, exist_ok=True)
            for name, model in models.items():
                print(f"Running the inference of unsupervised model using the engine - {name}")
                model.evaluate(x_test, y_test.values, output_dir=target_output_directory)
        except Exception as e:
            print(f" error during Processing the testing of unsupervised method on {name_of_target}: {e}")

if __name__ == "__main__":
    unsupervised_pipeline_with_unsw_as_source()