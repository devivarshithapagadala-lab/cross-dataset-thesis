from src.preprocessing.preprocessing import preprocess, load_csv_after_removing_duplicates
from src.models.isolation_forest import IsolationForestModel

def experiment_1():
    train_df = load_csv_after_removing_duplicates("data/CICIDS2017/Monday-WorkingHours.pcap_ISCX.csv").head(50000)
    x_train, y_train, trained_scaler = preprocess(train_df, scaler=None)
    test_df = load_csv_after_removing_duplicates("data/CSE-CIC-IDS2018/02-14-2018.csv").head(100000)
    x_test, y_test = preprocess(test_df, scaler=trained_scaler)
    isolation_forest = IsolationForestModel(contamination=0.05)
    isolation_forest.fit(x_train)
    isolation_forest.evaluate(x_test, y_test.values, output_dir="results/studies/experiments/exp1_saturation_leakage")

if __name__ == "__main__":
    experiment_1()