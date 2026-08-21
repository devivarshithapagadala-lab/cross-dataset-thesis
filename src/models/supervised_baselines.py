import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupervisedModels:
    def __init__(self, type_of_the_model="Random_Forest"):
        self.type_of_the_model = type_of_the_model
        if type_of_the_model == "Random_Forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif type_of_the_model == "Neural_Network":
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        elif type_of_the_model == "Gradient_Boosting":
            self.model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
        elif type_of_the_model == "Logistic_Regression":
            self.model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"The given supervised model is not known- {type_of_the_model}")

    def fit(self, x_train, y_train, **kwargs):
        logger.info(f"Training the Supervised architecture of the model {self.type_of_the_model} ")
        self.model.fit(x_train, np.ravel(y_train))
        return self
    def predict(self, x_test):
        return self.model.predict(x_test)
    def evaluate(self, x_test, y_test, output_dir=None, **kwargs):
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        cnfsn_mtrx = confusion_matrix(y_test, predictions)
        report_dictionary = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        report_string = classification_report(y_test, predictions, zero_division=0)
        print(f"\n Results of the model {self.type_of_the_model.upper()}")
        print(f"score of Accuracy: {accuracy*100:.2f}%\n")
        print("Confusion Matrix-")
        print(cnfsn_mtrx)
        print("\nReport of the classification-")
        print(report_string)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            clean_name = self.type_of_the_model.lower()
            df_metrics = pd.DataFrame(report_dictionary).transpose()
            df_metrics.to_csv(os.path.join(output_dir, f"{clean_name}_metrics.csv"))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cnfsn_mtrx, annot=True, fmt="d", cmap="Reds",xticklabels=["BENIGN", "ATTACK"],yticklabels=["BENIGN", "ATTACK"])
            plt.title(f"Confusion Matrix of the Supervised model {self.type_of_the_model.replace('_', ' ')}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{clean_name}_confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
        return predictions