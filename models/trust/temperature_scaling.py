import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU
from torch.utils.data import DataLoader

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            
    return ece, bin_accs, bin_confs

def plot_reliability_diagram(y_true, probs_uncal, probs_cal, save_path):
    ece_uncal, accs_uncal, confs_uncal = expected_calibration_error(y_true, probs_uncal)
    ece_cal, accs_cal, confs_cal = expected_calibration_error(y_true, probs_cal)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.plot(confs_uncal, accs_uncal, 's-', label="Uncalibrated")
    plt.xlabel("Mean predicted probability (Confidence)")
    plt.ylabel("Fraction of positives (Accuracy)")
    plt.title(f"Before Calibration\\nECE = {ece_uncal:.4f}")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.plot(confs_cal, accs_cal, 's-', color='green', label="Calibrated")
    plt.xlabel("Mean predicted probability (Confidence)")
    plt.ylabel("Fraction of positives (Accuracy)")
    plt.title(f"After Temperature Scaling\\nECE = {ece_cal:.4f}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def optimize_temperature():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    data_dir = os.path.join(base_dir, 'data', 'processed')
    val_df = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    
    W, B = 12, 128
    val_ds = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    val_loader = DataLoader(val_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=val_ds.F_node, edge_dim=val_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    edge_index = val_ds.edge_index.to(device)
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for xb_n, xb_e, _, yb in val_loader:
            xb_n, xb_e = xb_n.to(device), xb_e.to(device)
            logits = model(xb_n, xb_e, edge_index).squeeze(-1)
            all_logits.append(logits.cpu())
            all_labels.append(yb)
            
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    
    # Calculate uncalibrated metrics
    probs_uncal = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    
    nll_uncal = log_loss(labels_np, probs_uncal)
    brier_uncal = brier_score_loss(labels_np, probs_uncal)
    ece_uncal, _, _ = expected_calibration_error(labels_np, probs_uncal)
    
    # Optimize Temperature via Grid Search (Robust 1D optimization)
    best_nll = float('inf')
    opt_T = 1.0
    for t in np.arange(0.1, 5.0, 0.01):
        cal_probs = torch.sigmoid(logits / t).numpy()
        nll = log_loss(labels_np, cal_probs)
        if nll < best_nll:
            best_nll = nll
            opt_T = t
    
    # Calculate calibrated metrics
    calibrated_logits = logits / opt_T
    probs_cal = torch.sigmoid(calibrated_logits).detach().numpy()
    
    nll_cal = log_loss(labels_np, probs_cal)
    brier_cal = brier_score_loss(labels_np, probs_cal)
    ece_cal, _, _ = expected_calibration_error(labels_np, probs_cal)
    
    print("--- Temperature Scaling (Validation Set) ---")
    print(f"Optimal Temperature (T): {opt_T:.4f}")
    print(f"NLL   | Before: {nll_uncal:.4f} -> After: {nll_cal:.4f}")
    print(f"Brier | Before: {brier_uncal:.4f} -> After: {brier_cal:.4f}")
    print(f"ECE   | Before: {ece_uncal:.4f} -> After: {ece_cal:.4f}")
    
    res = {
        "Optimal_Temperature": round(opt_T, 4),
        "NLL_Uncalibrated": round(nll_uncal, 4),
        "NLL_Calibrated": round(nll_cal, 4),
        "Brier_Uncalibrated": round(brier_uncal, 4),
        "Brier_Calibrated": round(brier_cal, 4),
        "ECE_Uncalibrated": round(ece_uncal, 4),
        "ECE_Calibrated": round(ece_cal, 4)
    }
    
    with open(os.path.join(base_dir, 'experiments', 'final', 'temperature.json'), 'w') as f:
        json.dump(res, f, indent=4)
        
    plot_path = os.path.join(base_dir, 'experiments', 'final', 'reliability_diagram.png')
    plot_reliability_diagram(labels_np, probs_uncal, probs_cal, plot_path)
    print(f"Saved reliability diagram to {plot_path}")

if __name__ == '__main__':
    optimize_temperature()
