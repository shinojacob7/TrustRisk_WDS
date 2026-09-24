import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import sys
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU

class GNNWrapper(nn.Module):
    def __init__(self, base_model, edge_index, temperature=1.0):
        super().__init__()
        self.base_model = base_model
        self.edge_index = edge_index
        self.temperature = temperature
        
    def forward(self, xb_n, xb_e):
        # Forward pass returning calibrated probabilities
        logits = self.base_model(xb_n, xb_e, self.edge_index)
        return torch.sigmoid(logits / self.temperature)

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x)) if np.mean(x) > 0 else 0.0

def run_explanation_layer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
    with open(os.path.join(base_dir, 'experiments', 'final', 'temperature.json'), 'r') as f:
        t_cfg = json.load(f)
    opt_T = t_cfg['Optimal_Temperature']
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    W = 12
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    test_ds = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=True)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=test_ds.F_node, edge_dim=test_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    edge_index = test_ds.edge_index.to(device)
    wrapper = GNNWrapper(model, edge_index, opt_T).to(device)
    
    # 1. Prepare Background Data for SHAP (using a small sample of normal training data)
    bg_loader = DataLoader(train_ds, batch_size=100, shuffle=True)
    bg_n, bg_e, _, _ = next(iter(bg_loader))
    bg_n, bg_e = bg_n.to(device), bg_e.to(device)
    
    explainer = shap.GradientExplainer(wrapper, [bg_n, bg_e])
    
    # 2. Get predictions using batched loader to avoid OOM
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_cal_probs = []
    with torch.no_grad():
        for xb_n, xb_e, _, _ in test_loader:
            probs = wrapper(xb_n.to(device), xb_e.to(device)).squeeze(-1).cpu().numpy()
            all_cal_probs.extend(probs)
            
    cal_probs = np.array(all_cal_probs)
    preds = (cal_probs >= 0.86).astype(int) # Using threshold from previous run
    
    # We will explain up to 50 positive predictions to save time
    pos_idx = np.where(preds == 1)[0][:50]
    
    if len(pos_idx) == 0:
        print("No positive predictions to explain. Using highest probability samples.")
        pos_idx = np.argsort(cal_probs)[-50:]
        
    print(f"Generating SHAP values iteratively for {len(pos_idx)} predicted events...")
    
    # Bypass PyTorch CuDNN RNN backward-in-eval-mode bug
    torch.backends.cudnn.enabled = False
    
    ecp_scores = []
    
    for idx in pos_idx:
        # Extract individual tensor
        n, e, _, _ = test_ds[idx]
        n_in = n.unsqueeze(0).to(device)
        e_in = e.unsqueeze(0).to(device)
        
        # Calculate SHAP for this single instance to prevent OOM
        shap_vals = explainer.shap_values([n_in, e_in])
        shap_n, shap_e = shap_vals[0], shap_vals[1]
        
        # Aggregate absolute SHAP values across time window and feature dimensions
        # Node importance: (Nodes,)
        node_imp = np.abs(shap_n[0]).sum(axis=(0, 2))
        # Edge importance: (Edges,)
        edge_imp = np.abs(shap_e[0]).sum(axis=(0, 2))
        
        # Total spatial importance array (19 nodes + 35 edges)
        spatial_imp = np.concatenate([node_imp, edge_imp])
        
        # Calculate Explanation Concentration Proxy (Gini)
        ecp = gini_coefficient(spatial_imp)
        ecp_scores.append(ecp)
        
    avg_ecp = np.mean(ecp_scores)
    
    print("--- SHAP Explanation Layer ---")
    print(f"Processed {len(pos_idx)} instances.")
    print(f"Mean Explanation Concentration Proxy (ECP): {avg_ecp:.4f}")
    
    np.save(os.path.join(base_dir, 'experiments', 'final', 'shap_ecp.npy'), ecp_scores)
    
    with open(os.path.join(base_dir, 'experiments', 'final', 'shap_metrics.json'), 'w') as f:
        json.dump({"Mean_ECP": round(avg_ecp, 4)}, f, indent=4)
        
    print("Saved ECP scores to experiments/final/shap_ecp.npy")

if __name__ == '__main__':
    run_explanation_layer()
