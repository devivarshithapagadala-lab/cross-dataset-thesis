from typing import Any, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_csv_after_removing_duplicates(path_of_the_file: str) -> pd.DataFrame:
    df = pd.read_csv(path_of_the_file)
    df.columns = df.columns.str.strip()
    rows_before_removal = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    rows_after_removal = len(df)
    removed = rows_before_removal - rows_after_removal
    percentage_of_removed = (removed / rows_before_removal * 100) if rows_before_removal > 0 else 0
    print(f"  {path_of_the_file} loaded the rows {rows_before_removal}, removed {removed} duplicates ({percentage_of_removed:.2f}%), {rows_after_removal} remaining")
    return df

features_that_are_common = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd PSH Flags', 'Fwd Header Length',
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


def preprocess(df: object, scaler: object = None,
               tag_of_the_dataset: object = "CIC") -> Union[tuple[Any, Any, StandardScaler], tuple[Any, Any]]:
    df.columns = df.columns.str.strip()
    if tag_of_the_dataset == "UNSW":
        # schema alignment of the dataset UNSW-NB15
        transformed_df = pd.DataFrame(index=df.index)
        transformed_df['Flow Duration'] = df['dur'] * 1000000.0  # Convert seconds to microseconds
        transformed_df['Total Fwd Packets'] = df['spkts']
        transformed_df['Total Backward Packets'] = df['dpkts']
        transformed_df['Fwd Packet Length Max'] = df['sbytes'] / df['spkts'].replace(0, 1)
        transformed_df['Fwd Packet Length Min'] = 0.0
        transformed_df['Fwd Packet Length Mean'] = df['sbytes'] / df['spkts'].replace(0, 1)
        transformed_df['Bwd Packet Length Max'] = df['dbytes'] / df['dpkts'].replace(0, 1)
        transformed_df['Bwd Packet Length Min'] = 0.0
        transformed_df['Bwd Packet Length Mean'] = df['dbytes'] / df['dpkts'].replace(0, 1)
        transformed_df['Flow Bytes/s'] = (df['sbytes'] + df['dbytes']) / df['dur'].replace(0, 1)
        transformed_df['Flow Packets/s'] = (df['spkts'] + df['dpkts']) / df['dur'].replace(0, 1)
        transformed_df['Flow IAT Mean'] = (df['dur'] * 1000000.0) / (df['spkts'] + df['dpkts']).replace(0, 1)
        transformed_df['Flow IAT Std'] = 0.0
        transformed_df['Flow IAT Max'] = df['dur'] * 1000000.0
        transformed_df['Flow IAT Min'] = 0.0
        transformed_df['Fwd PSH Flags'] = 0.0
        transformed_df['Fwd Header Length'] = df['spkts'] * 20.0
        transformed_df['Bwd Header Length'] = df['dpkts'] * 20.0
        transformed_df['Fwd Packets/s'] = df['spkts'] / df['dur'].replace(0, 1)
        transformed_df['Bwd Packets/s'] = df['dpkts'] / df['dur'].replace(0, 1)
        transformed_df['Min Packet Length'] = 0.0
        transformed_df['Max Packet Length'] = transformed_df[['Fwd Packet Length Max', 'Bwd Packet Length Max']].max(axis=1)
        transformed_df['Packet Length Mean'] = (df['sbytes'] + df['dbytes']) / (df['spkts'] + df['dpkts']).replace(0, 1)
        transformed_df['Packet Length Std'] = 0.0
        transformed_df['SYN Flag Count'] = 0.0
        transformed_df['RST Flag Count'] = 0.0
        transformed_df['ACK Flag Count'] = 0.0
        transformed_df['Idle Mean'] = 0.0
        transformed_df['Idle Std'] = 0.0
        transformed_df['Idle Max'] = 0.0
        transformed_df['Idle Min'] = 0.0
        x = transformed_df[features_that_are_common].copy()
        y = df['label'].copy()
    else:
        # schema alignment of the datasets CICIDS2017/CSE-CIC-IDS2018
        mapping_of_the_schema = {
            'Tot Fwd Pkts': 'Total Fwd Packets', 'Tot Bwd Pkts': 'Total Backward Packets',
            'Fwd Pkt Len Max': 'Fwd Packet Length Max', 'Fwd Pkt Len Min': 'Fwd Packet Length Min',
            'Fwd Pkt Len Mean': 'Fwd Packet Length Mean', 'Bwd Pkt Len Max': 'Bwd Packet Length Max',
            'Bwd Pkt Len Min': 'Bwd Packet Length Min', 'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
            'Flow Byts/s': 'Flow Bytes/s', 'Flow Pkts/s': 'Flow Packets/s',
            'Fwd Header Len': 'Fwd Header Length', 'Bwd Header Len': 'Bwd Header Length',
            'Fwd Pkts/s': 'Fwd Packets/s', 'Bwd Pkts/s': 'Bwd Packets/s',
            'Pkt Len Min': 'Min Packet Length', 'Pkt Len Max': 'Max Packet Length',
            'Pkt Len Mean': 'Packet Length Mean', 'Pkt Len Std': 'Packet Length Std',
            'SYN Flag Cnt': 'SYN Flag Count', 'RST Flag Cnt': 'RST Flag Count',
            'ACK Flag Cnt': 'ACK Flag Count'
        }
        df = df.rename(columns=mapping_of_the_schema)
        x = df[features_that_are_common].copy()
        y = df["Label"].apply(lambda x: 0 if str(x).lower() in ["benign", "normal"] else 1)
    x = x.replace([float("inf"), float("-inf")], np.nan)
    x = x.fillna(0)
    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        return x_scaled, y, scaler
    else:
        x_scaled = scaler.transform(x)
        return x_scaled, y