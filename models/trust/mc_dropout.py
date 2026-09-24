import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU

def enable_dropout(model):
    """Enables dropout layers during evaluation."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def run_mc_dropout(num_passes=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load optimal config and temperature
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
    with open(os.path.join(base_dir, 'experiments', 'final', 'temperature.json'), 'r') as f:
        t_cfg = json.load(f)
    opt_T = t_cfg['Optimal_Temperature']
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    test_df = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    W, B = 12, 128
    test_ds = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=True)
    test_loader = DataLoader(test_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=test_ds.F_node, edge_dim=test_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Standard Eval mode, but enable dropout
    model.eval()
    enable_dropout(model)
    
    edge_index = test_ds.edge_index.to(device)
    
    mc_preds = []
    all_labels = []
    
    print(f"Running {num_passes} stochastic forward passes for Epistemic Uncertainty...")
    
    with torch.no_grad():
        for pass_idx in range(num_passes):
            pass_probs = []
            pass_labels = []
            for xb_n, xb_e, _, yb in test_loader:
                xb_n, xb_e = xb_n.to(device), xb_e.to(device)
                logits = model(xb_n, xb_e, edge_index).squeeze(-1)
                
                # Apply Temperature Scaling!
                cal_logits = logits / opt_T
                probs = torch.sigmoid(cal_logits)
                
                pass_probs.append(probs.cpu())
                if pass_idx == 0:
                    pass_labels.append(yb)
                    
            mc_preds.append(torch.cat(pass_probs).numpy())
            if pass_idx == 0:
                all_labels = torch.cat(pass_labels).numpy()
                
    mc_preds = np.array(mc_preds)  # Shape: (num_passes, num_samples)
    
    # Calculate Epistemic Uncertainty (Predictive Variance)
    mean_probs = np.mean(mc_preds, axis=0)
    pred_variance = np.var(mc_preds, axis=0)
    
    # Separate into Normal vs Attack
    normal_vars = pred_variance[all_labels == 0]
    attack_vars = pred_variance[all_labels == 1]
    
    print("--- MC Dropout Epistemic Uncertainty ---")
    print(f"Mean Variance (Normal) : {np.mean(normal_vars):.6f}")
    print(f"Mean Variance (Attack) : {np.mean(attack_vars):.6f}")
    print(f"Max Variance (Normal)  : {np.max(normal_vars):.6f}")
    print(f"Max Variance (Attack)  : {np.max(attack_vars):.6f}")
    
    # Plot Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(normal_vars, bins=50, alpha=0.5, label='Normal Samples', density=True, color='blue')
    plt.hist(attack_vars, bins=50, alpha=0.5, label='Attack Samples', density=True, color='red')
    plt.xlabel('Epistemic Uncertainty (Predictive Variance)')
    plt.ylabel('Density')
    plt.title('Distribution of Epistemic Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(base_dir, 'experiments', 'final', 'epistemic_uncertainty.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    # Save raw array data for Trust Score integration later
    np.save(os.path.join(base_dir, 'experiments', 'final', 'mc_mean_probs.npy'), mean_probs)
    np.save(os.path.join(base_dir, 'experiments', 'final', 'mc_epistemic_var.npy'), pred_variance)
    
    print(f"Saved Epistemic Uncertainty distribution to {plot_path}")

if __name__ == '__main__':
    run_mc_dropout()
