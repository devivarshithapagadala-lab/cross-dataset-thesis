import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

datasets = { "UNSW_dataset": "data/UNSW_NB15/UNSW_NB15_training-set.csv",
            "CIC2017_dataset": "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv",
            "CIC2018_dataset": "data/CSE-CIC-IDS2018/02-14-2018.csv" }
os.makedirs("results/eda/descriptive_statistics", exist_ok=True )
for name, path in datasets.items():
    df = pd.read_csv(path)
    stats = df.describe(include="all")
    stats.to_csv( f"results/eda/descriptive_statistics/{name}_statistics.csv" )