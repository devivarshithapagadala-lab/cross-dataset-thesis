import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class IsolationForestModel:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination,random_state=42,n_jobs=-1)
    def fit(self, x_train: np.ndarray):
        print("Training the model 'Isolation Forest'")
        self.model.fit(x_train)
        print("Training of Isolation Forest has been done")
    def evaluate(self, x_test: np.ndarray, y_true: np.ndarray, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        print(" Running the inference of the outlier on the test data ")
        predictions_that_are_raw = self.model.predict(x_test)
        predictions = np.where(predictions_that_are_raw == -1, 1, 0)
        print("\n Results of the model 'isolation forest'")
        print(f"score of accuracy: {accuracy_score(y_true, predictions):.2%}")
        cnfsn_mtrx = confusion_matrix(y_true, predictions)
        report = classification_report(y_true,predictions,target_names=["BENIGN", "ATTACK"],zero_division=0,output_dict=True)
        df_metrics = pd.DataFrame(report).transpose()
        df_metrics.to_csv(os.path.join(output_dir,"isolation_forest_metrics.csv"))
        print("\nConfusion Matrix:")
        print(cnfsn_mtrx)
        print("\nReport of classification-")
        print(classification_report(y_true, predictions, target_names=['BENIGN', 'ATTACK'], zero_division=0))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cnfsn_mtrx, annot=True, fmt='d', cmap='Blues',xticklabels=['Predicted BENIGN', 'Predicted ATTACK'],yticklabels=['Actual BENIGN', 'Actual ATTACK'])
        plt.title('Confusion Matrix of cross dataset for the model isolation forest ')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "isolation_forest_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()