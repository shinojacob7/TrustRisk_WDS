import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import random
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU, point_adjust, get_probs

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_exp_A():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    print(f"Training Exp A (Class-Weighted BCE) on {device}")
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    
    W, B = 12, 64
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    val_ds   = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    
    edge_index = train_ds.edge_index
    e_idx_device = edge_index.to(device)

    labels_all = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    
    # Calculate pos_weight strictly on Training Data
    pos_w = n_n / n_a if n_a > 0 else 1.0
    print(f"Calculated pos_weight from training data: {pos_w:.2f} ({n_n} neg / {n_a} pos)")
    
    # Removed WeightedRandomSampler to purely test pos_weight in loss function
    train_loader = DataLoader(train_ds, batch_size=B, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=train_ds.F_node, edge_dim=train_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_score = 0.0
    patience, stall = 10, 0
    save_path = os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'exp_A_model.pth')

    for ep in range(1, 41):
        print(f"Epoch {ep}/40...")
        model.train()
        for xb_n, xb_e, _, yb in train_loader:
            xb_n, xb_e, yb = xb_n.to(device), xb_e.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb_n, xb_e, e_idx_device).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_labels, val_probs = get_probs(model, val_loader, device, edge_index)
        
        # Select best model based on PR-AUC or F1, using PR-AUC as stable metric
        val_pr_auc = average_precision_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0

        if val_pr_auc > best_val_score:
            best_val_score = val_pr_auc
            stall = 0
            torch.save(model.state_dict(), save_path)
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    print("\n--- Experiment A: Validation Results ---")
    model.load_state_dict(torch.load(save_path))
    val_labels, val_probs = get_probs(model, val_loader, device, edge_index)

    # Threshold Optimization on VALIDATION ONLY
    best_f1, optimal_thr = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            optimal_thr = t

    final_preds = (val_probs >= optimal_thr).astype(int)
    final_pa_preds = point_adjust(val_labels, final_preds)
    
    val_prec = precision_score(val_labels, final_pa_preds, zero_division=0)
    val_rec = recall_score(val_labels, final_pa_preds, zero_division=0)
    val_f1 = f1_score(val_labels, final_pa_preds, zero_division=0)
    val_roc = roc_auc_score(val_labels, val_probs)
    val_pr = average_precision_score(val_labels, val_probs)

    print("EXPERIMENT: Class-Weighted BCE")
    print(f"configuration: pos_weight={pos_w:.2f}, model config best_config.json")
    print(f"validation precision: {val_prec:.4f}")
    print(f"validation recall: {val_rec:.4f}")
    print(f"validation F1: {val_f1:.4f}")
    print(f"validation PR-AUC: {val_pr:.4f}")
    print(f"validation ROC-AUC: {val_roc:.4f}")
    print(f"selected threshold: {optimal_thr:.4f}")

if __name__ == '__main__':
    train_exp_A()
