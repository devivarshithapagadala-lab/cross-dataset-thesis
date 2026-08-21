import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

datasets = { "UNSW_dataset": "data/UNSW_NB15/UNSW_NB15_training-set.csv",
             "CIC2017_dataset": "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
             "CIC2018_dataset": "data/CSE-CIC-IDS2018/02-14-2018.csv" }
os.makedirs("results/eda/duplicates", exist_ok=True )
rows = []
for name, path in datasets.items():
    df = pd.read_csv(path)
    rows.append({ "Name of the dataset": name, "Total_no_of_Rows": len(df), "No_of_Duplicates": df.duplicated().sum() })
pd.DataFrame(rows).to_csv("results/eda/duplicates/duplicate_summary.csv", index=False )