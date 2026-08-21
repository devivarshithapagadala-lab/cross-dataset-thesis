import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderForUnsupervisedModels
from src.models.one_class_svm import OneClassSVMModel
from src.models.local_outlier_factor import LocalOutlierFactorModel

def unsupervised_pipeline_cross_dataset():
    training_raw = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    training_sampled = training_raw.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train, y_train, trained_scaler = preprocess(training_sampled, scaler=None, tag_of_the_dataset="CIC")
    print(f" Normal Baseline Array, Shape of training set- {x_train.shape}")

    target_dataset_for_testing = {
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
        "Local_Outlier_Factor": LocalOutlierFactorModel(n_neighbors=20, contamination=0.10)
    }
    for name, model in models.items():
        if name == "Adaptive_Autoencoder":
            model.fit(x_train, y_train.values, epochs=5)
        else:
            model.fit(x_train)
    print("Training of the four unsupervised models is finished.")

    for name_of_target, config in target_dataset_for_testing.items():
        print(f"\nEvaluating the shift of Domain of {name_of_target.upper()}")
        try:
            testing_raw = load_csv_after_removing_duplicates(str(config["path"]))
            pool_of_benign = testing_raw[testing_raw['Label'].astype(str).str.strip().str.lower() == 'benign']
            pool_of_attack = testing_raw[testing_raw['Label'].astype(str).str.strip().str.lower() != 'benign']
            sample_of_b = pool_of_benign.sample(n=config["benign_samples"], random_state=42, replace=True)
            sample_of_a = pool_of_attack.sample(n=config["attack_samples"], random_state=42, replace=True)
            testing_sampled = pd.concat([sample_of_b, sample_of_a]).sample(frac=1, random_state=42).reset_index(drop=True)
            x_test, y_test = preprocess(testing_sampled, scaler=trained_scaler, tag_of_the_dataset=config["tag"])
            target_output_dir = os.path.join("results", "studies", "analysis","cross_dataset_unsupervised_2017_source")
            os.makedirs(target_output_dir, exist_ok=True)
            for name, model in models.items():
                print(f"Running the inference of unsupervised model using the engine: {name}")
                model.evaluate(x_test, y_test.values, output_dir=target_output_dir)
        except Exception as e:
            print(f" error during Processing the testing of unsupervised method on {name_of_target}: {e}")


if __name__ == "__main__":
    unsupervised_pipeline_cross_dataset()