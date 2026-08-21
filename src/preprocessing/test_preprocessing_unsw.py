import pandas as pd
from src.preprocessing.preprocessing import preprocess

def alignment_of_the_dataset_unsw():
    path_of_the_dataset_unsw = "data/UNSW_NB15/UNSW_NB15_training-set.csv"
    try:
        df = pd.read_csv(path_of_the_dataset_unsw)
        print(f" The Raw file of the dataset unsw has loaded successfully, Shape of the raw- {df.shape}")
        x, y, _ = preprocess(df, scaler=None, tag_of_the_dataset="UNSW")  # this is added to catch the scaler
        print("\nmetrics of post alignment")
        print(" shape of aligned feature matrix-", x.shape)
        print("shape of target label in form of array-", y.shape)
        print("\n mapped ground truth distribution which is unsupervised-")
        print(y.value_counts())
        print(" o indicates normal baseline traffic, 1 indicates anomalies of attack")
    except FileNotFoundError:
        print(f" File not found error- given file cannot be located at the given path - {path_of_the_dataset_unsw}")

if __name__ == "__main__":
    alignment_of_the_dataset_unsw()