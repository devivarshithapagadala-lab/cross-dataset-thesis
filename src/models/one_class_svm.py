import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

class OneClassSVMModel:
    def __init__(self, nu: float = 0.10):
        self.model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    def fit(self, x_train: np.ndarray):
        print("Training the model One Class SVM")
        self.model.fit(x_train)
        #here the boundary of one class svm is fixed
    def evaluate(self, x_test: np.ndarray, y_true: np.ndarray, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        print("Running the boundary inference of the model One Class SVM ")
        predictions_that_are_raw = self.model.predict(x_test)
        predictions = np.where(predictions_that_are_raw == -1, 1, 0)
        print("\nResults of the model one class svm")
        print(f"score of Accuracy- {accuracy_score(y_true, predictions):.2%}")
        cnfsn_mtrx = confusion_matrix(y_true, predictions)
        print("\nConfusion Matrix-")
        print(cnfsn_mtrx)
        print("\nReport of classification-")
        report = classification_report(y_true,predictions,target_names=['BENIGN', 'ATTACK'],zero_division=0,output_dict=True)
        print( classification_report(y_true,predictions,target_names=['BENIGN', 'ATTACK'],zero_division=0))
        pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir,"one_class_svm_metrics.csv"))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cnfsn_mtrx, annot=True, fmt='d', cmap='Greens',xticklabels=['Predicted BENIGN', 'Predicted ATTACK'],yticklabels=['Actual BENIGN', 'Actual ATTACK'])
        plt.title('Confusion Matrix of cross dataset for the model one class svm')
        plt.tight_layout()
        save_path = os.path.join(output_dir, "one_class_svm_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()