import pandas as pd
import os
list_of_datasets = { "UNSW-NB15_dataset": "data/UNSW_NB15/UNSW_NB15_training-set.csv",
                     "CICIDS2017_dataset": "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
                     "CSE-CIC-IDS2018_dataset": "data/CSE-CIC-IDS2018/02-14-2018.csv" }
os.makedirs("results/eda/dataset_inspection", exist_ok=True )
for name, path in list_of_datasets.items():
    print(f"Name of the dataset: {name}")
    try:
        df = pd.read_csv(path)
        print("\nShape of the dataset:")
        print(df.shape)
        print("\nColumns present in the dataset:")
        print(df.columns.tolist())
        print("\n1st five rows:")
        print(df.head())
        print("\nValues that are missing:")
        print(df.isnull().sum())

        # Columns that share similar labels
        for col in ["Label", "label"]:
            if col in df.columns:
                print("\nDistribution of the label:")
                print(df[col].value_counts())

        # Category columns of common attack
        for col in ["attack_cat", "Attack", "Category"]:
            if col in df.columns:
                print("\nCategories of the attack:")
                print(df[col].value_counts())

        summary = pd.DataFrame({
            "Rows": [df.shape[0]],
            "Columns": [df.shape[1]],
            "Missing": [df.isnull().sum().sum()]
        })
        summary.to_csv(f"results/eda/dataset_inspection/{name}_summary.csv",
            index=False )
    except Exception as e:
        print(f"There is some error in reading the dataset: {e}")