import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing.preprocessing import preprocess
from src.models.supervised_baselines import SupervisedModels

def in_domain_baseline_for_supervised_models_2017():
    print("Training and testing of supervised models on same dataset 'CICIDS2017'")
    raw = pd.read_csv("data/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv")
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
    print(f"\nshape of raw train data- {train_df.shape}")
    print(f"Shape of raw test data-  {test_df.shape}")
    x_train, y_train, scaler = preprocess(train_df, scaler=None, tag_of_the_dataset="CIC")
    x_test, y_test = preprocess(test_df, scaler=scaler, tag_of_the_dataset="CIC")
    print(f"\nTrain shape of the processed data- {x_train.shape}")
    print(f"Test shape of the Processed data-  {x_test.shape}")
    print(f"label distribution of train data- \n{y_train.value_counts()}")
    print(f"label distribution of test data\n{y_test.value_counts()}")
    models = {
        "Random_Forest": SupervisedModels(type_of_the_model="Random_Forest"),
        "Neural_Network": SupervisedModels(type_of_the_model="Neural_Network"),
        "Gradient_Boosting": SupervisedModels(type_of_the_model="Gradient_Boosting"),
        "Logistic_Regression": SupervisedModels(type_of_the_model="Logistic_Regression")
    }
    output_dir = "results/studies/analysis/in_domain_baseline_supervised_2017"
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        print(f"\nTraining the model {name} on the same dataset CICIDS2017")
        model.fit(x_train, y_train.values)
        print(f"Evaluation of the model {name} on the same dataset CICIDS2017")
        model.evaluate(x_test, y_test.values, output_dir=output_dir)
    print("\nBaseline of the in domain for supervised models have finished.")
if __name__ == "__main__":
    in_domain_baseline_for_supervised_models_2017()