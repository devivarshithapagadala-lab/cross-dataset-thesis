import pandas as pd

unsw_dataset = pd.read_csv("data/UNSW_NB15/UNSW_NB15_training-set.csv", nrows=1 )
cic17_dataset = pd.read_csv("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1 )
cic18_dataset = pd.read_csv("data/CSE-CIC-IDS2018/02-14-2018.csv", nrows=1 )

print("\nColumns present in the dataset UNSW:")
for c in unsw_dataset.columns:
    print(repr(c))

print("\nColumns present in the dataset CIC2017:")
for c in cic17_dataset.columns:
    print(repr(c))

print("\nColumns present in the dataset CIC2018:")
for c in cic18_dataset.columns:
    print(repr(c))