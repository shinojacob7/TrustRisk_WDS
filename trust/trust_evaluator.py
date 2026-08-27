import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import shap
from scipy.stats import entropy
import os

class TemperatureScaler(nn.Module):
    """
    Learns a Temperature parameter T on a validation set to calibrate probabilities.
    Instead of a hardcoded T=1.5, this uses NLL optimization.
    """
    def __init__(self, model, device):
        super(TemperatureScaler, self).__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Initialize at 1.5

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def calibrate(self, val_loader):
        """
        Optimize the temperature T using LBFGS on the validation set.
        """
        self.model.eval()
        nll_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for x, y in val_loader:
                logits_list.append(self.model(x.to(self.device)).squeeze())
                labels_list.append(y.to(self.device))
                
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        print(f"Optimal Temperature (T) calibrated to: {self.temperature.item():.4f}")
        return self.temperature.item()


class MCDropoutEvaluator:
    """
    Quantifies Epistemic Uncertainty using Monte Carlo Dropout.
    Provides Predictive Mean, Variance, and Entropy.
    """
    def __init__(self, model, num_samples=30):
        self.model = model
        self.num_samples = num_samples

    def evaluate(self, x):
        # Enable dropout during inference
        self.model.train() 
        
        stochastic_probs = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(x).squeeze()
                probs = torch.sigmoid(logits)
                stochastic_probs.append(probs.cpu().numpy())
                
        stochastic_probs = np.array(stochastic_probs)
        
        # Predictive Mean (Confidence)
        mean_prob = np.mean(stochastic_probs, axis=0)
        # Predictive Variance (Epistemic Uncertainty)
        variance = np.var(stochastic_probs, axis=0)
        
        # Predictive Entropy: - (p*log(p) + (1-p)*log(1-p))
        eps = 1e-10
        p = np.clip(mean_prob, eps, 1 - eps)
        predictive_entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
        
        return mean_prob, variance, predictive_entropy


class ExplanationReliability:
    """
    Formal mathematical definition of Explanation Reliability Score (ERS).
    Instead of a heuristic, we use the Normalized Information Entropy of the SHAP values.
    If the model relies heavily on a few specific sensors (decisive explanation), Entropy is low -> ERS is high.
    If the model relies uniformly on noise across all sensors, Entropy is high -> ERS is low.
    """
    def __init__(self, model, background_data):
        self.model = model.eval()
        self.explainer = shap.GradientExplainer(self.model, background_data)

    def compute_ers(self, x_instance):
        """
        x_instance: shape (1, seq_len, num_features)
        Returns the ERS scalar [0, 1] and the raw SHAP feature importances.
        """
        # SHAP requires requires_grad=True for GradientExplainer
        x_instance = x_instance.clone().detach().requires_grad_(True)
        
        shap_values = self.explainer.shap_values(x_instance)
        
        # Aggregate SHAP over the temporal window to get feature-level importance
        # shap_values shape: (1, seq_len, num_features)
        feature_importance = np.abs(shap_values).mean(axis=(0, 1))
        
        # Calculate Normalized Entropy of the feature importances
        # 1. Normalize to create a probability distribution
        total_importance = np.sum(feature_importance) + 1e-10
        p_dist = feature_importance / total_importance
        
        # 2. Calculate Shannon Entropy
        # Max entropy for K features is log(K)
        K = len(feature_importance)
        max_entropy = np.log(K)
        
        current_entropy = entropy(p_dist + 1e-10) # Add epsilon to avoid log(0)
        
        # 3. ERS = 1 - (Entropy / Max_Entropy)
        # Bounded between 0 (completely uniform/unreliable) and 1 (perfectly decisive)
        ers = 1.0 - (current_entropy / max_entropy)
        
        return ers, feature_importance

if __name__ == "__main__":
    print("Trust & Reliability Engine Modules (Phase 4) defined successfully.")
    print("- TemperatureScaler (Probability Calibration via NLL)")
    print("- MCDropoutEvaluator (Uncertainty Quantification)")
    print("- ExplanationReliability (Formal ERS via Normalized SHAP Entropy)")
