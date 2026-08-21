import os
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.supervised_baselines import SupervisedModels


def supervised_pipeline_with_unsw_as_source():
    training_raw = load_csv_after_removing_duplicates("data/UNSW_NB15/UNSW_NB15_training-set.csv")
    training_sampled = training_raw.sample(n=min(50000, len(training_raw)), random_state=42).reset_index(drop=True)
    x_train, y_train, trained_scaler = preprocess(training_sampled, scaler=None, tag_of_the_dataset="UNSW")
    print(f" Labeled Training Array, Shape of training set- {x_train.shape}")

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
        "Random_Forest": SupervisedModels(type_of_the_model="Random_Forest"),
        "Neural_Network": SupervisedModels(type_of_the_model="Neural_Network"),
        "Gradient_Boosting": SupervisedModels(type_of_the_model="Gradient_Boosting"),
        "Logistic_Regression": SupervisedModels(type_of_the_model="Logistic_Regression")
    }
    for name, model in models.items():
        model.fit(x_train, y_train.values)
    print("Training of the four supervised models is finished.")

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
            target_output_directory = os.path.join("results", "studies", "analysis", "cross_dataset_supervised_unsw_source")
            os.makedirs(target_output_directory, exist_ok=True)
            for name, model in models.items():
                print(f"Running the inference of supervised model using the engine - {name}")
                model.evaluate(x_test, y_test.values, output_dir=target_output_directory)
        except Exception as e:
            print(f" error during Processing the testing of supervised method on {name_of_target}: {e}")

if __name__ == "__main__":
    supervised_pipeline_with_unsw_as_source()