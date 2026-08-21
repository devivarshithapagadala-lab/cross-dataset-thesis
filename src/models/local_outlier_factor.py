import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalOutlierFactorModel:
    def __init__(self, n_neighbors=20, contamination=0.10):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors,contamination=self.contamination,novelty=True,n_jobs=-1)
    def fit(self, x_train, *args, **kwargs):
        logger.info(f"Fitting the model Local outlier factor with n_neighbors={self.n_neighbors}...")
        self.model.fit(x_train)
        return self
    def predict(self, x_test):
        predictions_that_are_raw = self.model.predict(x_test)
        regularized_predictions = np.where(predictions_that_are_raw == 1, 0, 1)
        return regularized_predictions
    def evaluate(self, x_test, y_test, output_dir=None, **kwargs):
        predictions = self.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        cnfsn_mtrx = confusion_matrix(y_test, predictions)
        report_dictionary = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_string = classification_report(y_test, predictions, zero_division=0)
        print(f"\n Results of the model Local Outlier Factor")
        print(f"score of Accuracy: {acc*100:.2f}%\n")
        print("Confusion Matrix-")
        print(cnfsn_mtrx)
        print("\nReport of classification-")
        print(report_string)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df_metrics = pd.DataFrame(report_dictionary).transpose()
            df_metrics.to_csv(os.path.join(output_dir, "local_outlier_factor_metrics.csv"))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cnfsn_mtrx, annot=True, fmt="d", cmap="Blues",xticklabels=["BENIGN", "ATTACK"],yticklabels=["BENIGN", "ATTACK"])
            plt.title("Confusion Matrix of the model Local Outlier Factor")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "local_outlier_factor_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
        return predictions