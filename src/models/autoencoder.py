import os
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def defining_the_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class NetOfAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential( nn.Linear(input_dim, 16),nn.ReLU(),nn.Linear(16, 8),nn.ReLU())
        self.decoder = nn.Sequential( nn.Linear(8, 16),nn.ReLU(),nn.Linear(16, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))
class AutoencoderForUnsupervisedModels:
    def __init__(self, input_dim: int, lr: float = 0.005, seed: int = 42):
        defining_the_seed(seed=seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NetOfAutoencoder(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 5, batch_size: int = 512):
        print ("To profile the neural reconstruction, benign vecotrs are filtered here")
        data_related_to_benign = x_train[y_train == 0]
        tensor_of_x = torch.tensor(data_related_to_benign, dtype=torch.float32).to(self.device)
        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensor_of_x),
            batch_size=batch_size,
            shuffle=True
        )
        self.model.train()
        print(f" Optimization of the weights of autoencoder over the {epochs} epochs ")
        for epoch in range(epochs):
            overall_loss = 0
            for batch in data_loader:
                inputs = batch[0]
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                overall_loss += loss.item() * inputs.size(0)
            print(f" Epoch {epoch+1}/{epochs}, Mse Loss of reconstruction: {overall_loss / len(data_related_to_benign):.6f}")

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray, output_dir: str = "results"):
        os.makedirs(output_dir, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            tensor_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
            reconstructions = self.model(tensor_test)
            errors = torch.mean((tensor_test - reconstructions) ** 2, dim=1).cpu().numpy()
            benign_errors = errors[y_test == 0]
            threshold = np.mean(benign_errors) + (3 * np.std(benign_errors))
        predictions = np.where(errors > threshold, 1, 0)
        print("\n Results of adaptive autoencoder for pytorch ")
        print(f" Evaluated adaptive mathematical line of threshold- {threshold:.6f}")
        print(f"Score of ROC-AUC- {roc_auc_score(y_test, errors):.4f}")
        cnfsn_mtrx = confusion_matrix(y_test, predictions)
        report = classification_report( y_test,predictions,target_names=["BENIGN", "ATTACK"], zero_division=0,output_dict=True)
        print( classification_report(y_test,predictions,target_names=["BENIGN", "ATTACK"],zero_division=0))
        pd.DataFrame(report).transpose().to_csv(os.path.join( output_dir,"autoencoder_metrics.csv" ))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cnfsn_mtrx, annot=True, fmt='d', cmap='Oranges',xticklabels=['Predicted BENIGN', 'Predicted ATTACK'],yticklabels=['Actual BENIGN', 'Actual ATTACK'])
        plt.title('Confusion Matrix of cross dataset for the model autoencoder')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "autoencoder_confusion_matrix.png"), dpi=300)
        plt.close()
        plt.figure(figsize=(8, 5))
        sns.histplot(data=pd.DataFrame({'Error': errors, 'Class': y_test}),x='Error', hue='Class', bins=50, kde=True, palette='Set1', element='step')
        plt.axvline(x = threshold, color='red', linestyle='--', label=f'Adaptive Threshold ({threshold:.4f})')
        plt.title('Profile of reconstruction error distribution for the model autoencoder')
        plt.xlabel('MSE - mean squared error')
        plt.ylabel('Frequency of the count of Packet')
        plt.legend(labels=[f'Adaptive Threshold ({threshold:.4f})', 'ATTACK', 'BENIGN'])
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "autoencoder_error_distribution.png"), dpi=300)
        plt.close()

    def evaluation_of_threshold(self, x_test, y_test, type_of_threshold="mean_3std"):
        self.model.eval()
        with torch.no_grad():
            test_of_tensor = torch.tensor( x_test, dtype=torch.float32).to(self.device)
            reconstruction = self.model(test_of_tensor)
            errors = torch.mean((test_of_tensor - reconstruction) ** 2,dim=1).cpu().numpy()
        errors_related_to_benign = errors[y_test == 0]
        mean = np.mean(errors_related_to_benign)
        std = np.std(errors_related_to_benign)
        if type_of_threshold == "mean_2std":
            threshold = mean + 2 * std
        elif type_of_threshold == "mean_3std":
            threshold = mean + 3 * std
        elif type_of_threshold == "mean_4std":
            threshold = mean + 4 * std
        elif type_of_threshold.startswith("percentile"):
            value_of_percentile = float(type_of_threshold.replace("percentile", ""))
            threshold = np.percentile(errors_related_to_benign, value_of_percentile)
        else:
            raise ValueError("Type of the threshold is not known")
        predictions = np.where(errors > threshold, 1, 0)
        report = classification_report( y_test,predictions,output_dict=True,zero_division=0)
        return {
            "Threshold": threshold,
            "Accuracy": report["accuracy"],
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1": report["1"]["f1-score"],
            "ROC_AUC": roc_auc_score(y_test, errors)
        }