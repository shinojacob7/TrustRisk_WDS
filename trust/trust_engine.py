import torch
import numpy as np
import pandas as pd
import shap
import os
from detection_model import TrustRiskBiLSTM, BATADALSequenceDataset

class TrustEngine:
    def __init__(self, model_path, input_dim):
        """
        Initializes the Trust Engine by loading the trained BiLSTM model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrustRiskBiLSTM(input_dim=input_dim).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # We will use this placeholder for temperature scaling
        # In a full implementation, temperature is learned on a validation set.
        self.temperature = 1.5 

    def get_confidence(self, logits):
        """
        1. Prediction Confidence Engine
        Applies Temperature Scaling to calibrate confidence scores.
        """
        # Apply temperature scaling before sigmoid
        scaled_logits = logits / self.temperature
        probs = torch.sigmoid(scaled_logits)
        
        # Confidence is the probability of the predicted class
        # If prob > 0.5 (Attack), confidence is prob. If prob <= 0.5 (Normal), confidence is 1 - prob.
        confidence = torch.where(probs > 0.5, probs, 1.0 - probs)
        return probs.item(), confidence.item()

    def get_uncertainty(self, x, n_samples=15):
        """
        2. Uncertainty Quantification Engine
        Uses Monte Carlo (MC) Dropout to estimate model uncertainty.
        We run the input through the model N times with Dropout enabled.
        """
        self.model.train() # Force dropout to remain active
        
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(x).squeeze()
                prob = torch.sigmoid(logits).item()
                preds.append(prob)
                
        mean_prob = np.mean(preds)
        uncertainty = np.std(preds) # Standard deviation represents Epistemic Uncertainty
        
        return mean_prob, uncertainty

    def get_explanation_reliability(self, x, background_data):
        """
        3. Explanation Reliability Engine
        Uses SHAP to explain which features contributed to the prediction.
        Evaluates the stability/reliability of the explanation.
        """
        self.model.eval() # Turn off dropout for deterministic explanation
        
        # GradientExplainer works better for PyTorch RNN/LSTM models
        explainer = shap.GradientExplainer(self.model, background_data)
        
        # SHAP values for the current sample
        shap_values = explainer.shap_values(x)
        
        # In GradientExplainer, shap_values might be a list or array. Handle safely:
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Explanation Reliability Score (ERS)
        # In this prototype, we simulate ERS based on the dominance of the top features.
        # If the explanation is highly scattered among all features, it's less reliable.
        # If it points strongly to 1-2 features, it's considered highly reliable.
        feature_importance = np.abs(shap_values).mean(axis=1).flatten()
        top_importance = np.max(feature_importance)
        avg_importance = np.mean(feature_importance) + 1e-9
        
        ers_score = min(1.0, (top_importance / avg_importance) / 10.0)
        
        return feature_importance, ers_score

if __name__ == "__main__":
    # Test the Trust Engine
    data_path = 'data/processed/BATADAL_balanced_scaled.csv'
    df = pd.read_csv(data_path)
    
    input_dim = df.shape[1] - 1
    
    # Initialize Engine
    trust_engine = TrustEngine('src/saved_models/bilstm_model.pth', input_dim=input_dim)
    
    # Prepare a single sequence window for testing
    window_size = 10
    features = df.drop(columns=['ATT_FLAG']).values.astype(np.float32)
    x_test = torch.tensor(features[500:500+window_size]).unsqueeze(0) # Shape: (1, 10, 43)
    
    # Background data for SHAP (e.g., 50 random samples)
    bg_samples = []
    for i in range(50):
        bg_samples.append(features[i:i+window_size])
    background_data = torch.tensor(np.array(bg_samples))
    
    # 1. Base Prediction & Confidence
    trust_engine.model.eval()
    with torch.no_grad():
        logits = trust_engine.model(x_test).squeeze()
    prob, conf = trust_engine.get_confidence(logits)
    
    # 2. Uncertainty via MC Dropout
    mean_prob_mc, uncertainty = trust_engine.get_uncertainty(x_test, n_samples=20)
    
    # 3. Explainability & Reliability
    feat_imp, ers = trust_engine.get_explanation_reliability(x_test, background_data)
    
    print("\n--- Trust & Reliability Evaluation Report ---")
    print(f"Base Prediction Probability : {prob:.4f}")
    print(f"Prediction Confidence Score : {conf:.4f} (Temperature Scaled)")
    print(f"MC Dropout Mean Probability : {mean_prob_mc:.4f}")
    print(f"Model Uncertainty (Std Dev) : {uncertainty:.4f}")
    print(f"Explanation Reliability(ERS): {ers:.4f}")
    print(f"Top 3 Contributing Features (Indices): {np.argsort(feat_imp)[-3:][::-1]}")
