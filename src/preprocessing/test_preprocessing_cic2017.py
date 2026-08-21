import pandas as pd
from src.preprocessing.preprocessing import preprocess

def preprocessing_test_of_dataset_cic2017():
    path = "data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv"
    try:
        df = pd.read_csv(path, nrows=50000)
        print(f" data reading is successful Raw Shape is- {df.shape}")
        x_scaled, y, trained_scaler = preprocess(df, scaler=None)
        print("\n metadata that is processed")
        print(" feature matrix shape of x scaled-", x_scaled.shape)
        print(" target shape of vector (y True)-", y.shape)
        print("\nClass distribution of label")
        print(y.value_counts())
        print("Testing of Preprocessing is finished")
    except FileNotFoundError:
        print(f"File not found, please Check the path: {path}")
    except Exception as e:
        print(f"There is an error while processing the file: {e}")
if __name__ == "__main__":
    preprocessing_test_of_dataset_cic2017()