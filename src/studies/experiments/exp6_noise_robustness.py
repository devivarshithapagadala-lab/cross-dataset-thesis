import os
import numpy as np
import pandas as pd
from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel
from src.models.one_class_svm import OneClassSVMModel
from src.models.autoencoder import AutoencoderForUnsupervisedModels

def addition_of_noise(X, noise_level):
    noise = np.random.normal(loc=0, scale=noise_level, size=X.shape)
    return X + noise

def experiment_6():
    np.random.seed(42)
    train = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv")
    train = train.sample(n=50000, random_state=42).reset_index(drop=True)
    x_train, y_train, scaler = preprocess(train, scaler=None)
    test = load_csv_after_removing_duplicates("data/CSE-CIC-IDS2018/02-14-2018.csv")
    benign = test[test["Label"].str.lower() == "benign"].sample(n=45000, random_state=42)
    attack = test[test["Label"].str.lower() != "benign"].sample(n=5000, random_state=42)
    test = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
    x_test, y_test = preprocess(test, scaler=scaler)

    print("\nTraining of the model Isolation Forest")
    isolation_forest = IsolationForestModel(contamination=0.10)
    isolation_forest.fit(x_train)
    print("\nTraining of the model One Class SVM")
    one_class_svm = OneClassSVMModel(nu=0.10)
    one_class_svm.fit(x_train)
    print("\nTraining of the model Adaptive Autoencoder")
    auto = AutoencoderForUnsupervisedModels(input_dim=x_train.shape[1])
    auto.fit(x_train, y_train.values, epochs=5)

    levels_of_the_noise = [0.00, 0.05, 0.10, 0.20, 0.30]
    for level in levels_of_the_noise:
        print(f"\nTesting the level of noise - {int(level*100)}%")
        x_noise = addition_of_noise(x_test, level)
        folder = (
            f"results/studies/experiments/"
            f"exp6_noise_robustness/noise_{int(level*100)}"
)
        os.makedirs(folder, exist_ok=True)
        print("\nIsolation Forest")
        isolation_forest.evaluate(x_noise, y_test.values, output_dir=folder)
        print("\nOne Class SVM")
        one_class_svm.evaluate(x_noise, y_test.values, output_dir=folder)
        print("\nAdaptive Autoencoder")
        auto.evaluate(x_noise, y_test.values, output_dir=folder)

if __name__ == "__main__":
    experiment_6()