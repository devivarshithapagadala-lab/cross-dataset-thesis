import os
import pandas as pd

datasets = { "UNSW_dataset": "data/UNSW_NB15/UNSW_NB15_training-set.csv",
             "CIC2017_dataset": "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
             "CIC2018_dataset": "data/CSE-CIC-IDS2018/02-14-2018.csv" }
os.makedirs("results/eda/missing_values", exist_ok=True )
for name, path in datasets.items():
    df = pd.read_csv(path)
    missing = pd.DataFrame({
        "Feature_of_the_dataset": df.columns,
        "no_of_Missing_values": df.isnull().sum().values
    })
    missing.to_csv(f"results/eda/missing_values/{name}_missing.csv", index=False )