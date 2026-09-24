import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, average_precision_score, confusion_matrix
)
import random
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU, get_event_metrics, get_probs, point_adjust

def apply_threshold(labels, probs, thr):
    raw_preds = (probs >= thr).astype(int)
    pa_preds = point_adjust(labels, raw_preds)
    f1   = f1_score(labels, pa_preds, zero_division=0)
    prec = precision_score(labels, pa_preds, zero_division=0)
    rec  = recall_score(labels, pa_preds, zero_division=0)
    acc  = accuracy_score(labels, pa_preds)
    cm = confusion_matrix(labels, pa_preds)
    
    # Use standard event metrics from base
    events = np.split(np.where(labels == 1)[0], np.where(np.diff(np.where(labels == 1)[0]) != 1)[0] + 1) if (labels==1).sum() > 0 else []
    det_ev = 0
    total_delay = 0
    for ev in events:
        if np.any(raw_preds[ev] == 1):
            det_ev += 1
            total_delay += np.argmax(raw_preds[ev] == 1)
            
    fp_mask = (raw_preds == 1) & (labels == 0)
    fp_indices = np.where(fp_mask)[0]
    fa = 0
    if len(fp_indices) > 0:
        fa = len(np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1))
        
    edr = det_ev / len(events) if events else 0.0
    m_delay = total_delay / det_ev if det_ev > 0 else 0.0
    
    return acc, f1, prec, rec, cm, edr, m_delay, det_ev, len(events), fa

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_final_model():
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    print(f"Training Final Model on {device} with config:")
    print(cfg)
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))

    W = 12
    B = 64
    
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    val_ds   = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    test_ds  = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=True)
    
    edge_index = train_ds.edge_index
    e_idx_device = edge_index.to(device)

    labels_all = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    wts = np.where(labels_all == 1, 1.0 / n_a, 1.0 / n_n)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.float64), len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=B, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=train_ds.F_node, edge_dim=train_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_auc = 0.0
    patience, stall = 10, 0
    save_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')

    for ep in range(1, 41):
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
        val_auc = roc_auc_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            stall = 0
            torch.save(model.state_dict(), save_path)
        else:
            stall += 1
            if stall >= patience:
                break

    print("\n--- Final Untouched Test Evaluation ---")
    model.load_state_dict(torch.load(save_path))
    val_labels, val_probs = get_probs(model, val_loader, device, edge_index)
    test_labels, test_probs = get_probs(model, test_loader, device, edge_index)

    best_f1, optimal_thr = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            optimal_thr = t

    print(f"Validation Threshold Selected: {optimal_thr:.4f}")

    t_acc, t_f1, t_prec, t_rec, cm, edr, m_delay, det_ev, tot_ev, fa = apply_threshold(test_labels, test_probs, optimal_thr)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_pr_auc = average_precision_score(test_labels, test_probs)

    print(f"Test Accuracy  (PA) : {t_acc:.4f}")
    print(f"Test F1-Score  (PA) : {t_f1:.4f}")
    print(f"Test Precision (PA) : {t_prec:.4f}")
    print(f"Test Recall    (PA) : {t_rec:.4f}")
    print(f"Test ROC-AUC        : {test_auc:.4f}")
    print(f"Test PR-AUC         : {test_pr_auc:.4f}")
    print(f"Event Detection Rate: {edr:.2f} ({det_ev}/{tot_ev})")
    print(f"Mean Det. Delay     : {m_delay:.2f} frames")
    print(f"False Alarms (Event): {fa}")

    cm_list = cm.tolist() if isinstance(cm, np.ndarray) else cm

    metrics = {
        "Accuracy_PA": round(t_acc, 4),
        "F1_Score_PA": round(t_f1, 4),
        "Precision_PA": round(t_prec, 4),
        "Recall_PA": round(t_rec, 4),
        "ROC_AUC": round(test_auc, 4),
        "PR_AUC": round(test_pr_auc, 4),
        "Event_Detection_Rate": round(edr, 4),
        "Detected_Events": det_ev,
        "Total_Events": tot_ev,
        "Mean_Detection_Delay": round(m_delay, 4),
        "False_Alarm_Events": fa,
        "Confusion_Matrix": cm_list,
        "Threshold": round(optimal_thr, 4)
    }
    
    with open(os.path.join(base_dir, 'experiments', 'final', 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    train_final_model()
