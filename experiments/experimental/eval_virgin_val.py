import os
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU, point_adjust, get_probs

def eval_virgin():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    data_dir = os.path.join(base_dir, 'data', 'processed')
    val_df = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    
    W, B = 12, 64
    val_ds = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    val_loader = DataLoader(val_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=val_ds.F_node, edge_dim=val_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    save_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')
    model.load_state_dict(torch.load(save_path))
    
    val_labels, val_probs = get_probs(model, val_loader, device, val_ds.edge_index)

    # Phase 6B: diagnostic sweep
    # Find thresh for F1, F2, Recall 0.8, Recall 0.9, Precision 0.9
    metrics_log = []
    best_f1, f1_thr = 0, 0.5
    best_f2, f2_thr = 0, 0.5
    r80_thr, r90_thr, p90_thr = -1, -1, -1
    r80_f1, r90_f1, p90_f1 = 0, 0, 0

    for t in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        p = precision_score(val_labels, pa_preds, zero_division=0)
        r = recall_score(val_labels, pa_preds, zero_division=0)
        f2 = (5 * p * r) / (4 * p + r + 1e-9)
        
        if f1 > best_f1: best_f1, f1_thr = f1, t
        if f2 > best_f2: best_f2, f2_thr = f2, t
        
        if r >= 0.80 and r80_thr == -1: r80_thr, r80_f1 = t, f1
        if r >= 0.90 and r90_thr == -1: r90_thr, r90_f1 = t, f1
        if p >= 0.90: p90_thr, p90_f1 = t, f1 # Will take the highest threshold that maintains p >= 0.90

    # Optimal threshold selected natively is 0.86
    t = 0.86
    preds = (val_probs >= t).astype(int)
    pa_preds = point_adjust(val_labels, preds)
    val_prec = precision_score(val_labels, pa_preds, zero_division=0)
    val_rec = recall_score(val_labels, pa_preds, zero_division=0)
    val_f1 = f1_score(val_labels, pa_preds, zero_division=0)
    val_roc = roc_auc_score(val_labels, val_probs)
    val_pr = average_precision_score(val_labels, val_probs)

    print("PHASE 6B: VIRGIN BASELINE VALIDATION SWEEP")
    print(f"Max F1: {best_f1:.4f} at thresh {f1_thr:.2f}")
    print(f"Max F2: {best_f2:.4f} at thresh {f2_thr:.2f}")
    print(f"Threshold for Recall >= 0.8: {r80_thr:.2f} (F1: {r80_f1:.4f})")
    print(f"Threshold for Recall >= 0.9: {r90_thr:.2f} (F1: {r90_f1:.4f})")
    print(f"Threshold for Precision >= 0.9: {p90_thr:.2f} (F1: {p90_f1:.4f})")
    
    print("\nEXPERIMENT: Virgin Baseline")
    print(f"configuration: WeightedRandomSampler")
    print(f"validation precision: {val_prec:.4f}")
    print(f"validation recall: {val_rec:.4f}")
    print(f"validation F1: {val_f1:.4f}")
    print(f"validation PR-AUC: {val_pr:.4f}")
    print(f"validation ROC-AUC: {val_roc:.4f}")
    print(f"selected threshold: {t}")

if __name__ == '__main__':
    eval_virgin()
