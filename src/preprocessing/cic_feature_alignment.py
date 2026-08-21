import pandas as pd

cic17_dataset = pd.read_csv("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv", nrows=1)
cic18_dataset = pd.read_csv("data/CSE-CIC-IDS2018/02-14-2018.csv", nrows=1)
columns_of_the_dataset_cic17 = set(c.strip() for c in cic17_dataset.columns)
columns_of_the_dataset_cic18 = set(c.strip() for c in cic18_dataset.columns)
common = columns_of_the_dataset_cic17.intersection(columns_of_the_dataset_cic18)
print("Features of the dataset CIC2017-", len(columns_of_the_dataset_cic17))
print("Features of the dataset CIC2018-", len(columns_of_the_dataset_cic18))
print("\nFeatures that are common")
for columns in sorted(common):
    print(columns)
print("\nTotal num of the features that are common-", len(common))