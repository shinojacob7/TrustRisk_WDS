import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score
)
import random
import sys
import time

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import point_adjust, get_probs, PhysicalGNNLayer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MaskedPhysicalGNN_GRU(nn.Module):
    def __init__(self, node_dim, edge_dim, gnn_hidden, gru_hidden, num_gnn_layers, num_gru_layers, dropout):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, gnn_hidden)
        self.gnn_layers = nn.ModuleList([
            PhysicalGNNLayer(gnn_hidden, edge_dim, gnn_hidden) for _ in range(num_gnn_layers)
        ])
        self.gru = nn.GRU(gnn_hidden, gru_hidden, num_layers=num_gru_layers, batch_first=True, dropout=dropout if num_gru_layers > 1 else 0)
        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_attr, edge_index):
        B, W, N, F = x.shape
        x_flat = x.view(B * W, N, F)
        e_flat = edge_attr.view(B * W, edge_attr.shape[2], edge_attr.shape[3])
        
        x_gnn = self.node_proj(x_flat)
        for gnn in self.gnn_layers:
            x_gnn = gnn(x_gnn, e_flat, edge_index)
            
        x_gnn = x_gnn.view(B, W, N, -1)
        
        obs_mask = x[:, :, :, 1:2] 
        masked_x = x_gnn * obs_mask
        sum_pooled = masked_x.sum(dim=2) 
        mask_sum = obs_mask.sum(dim=2).clamp(min=1.0) 
        x_pooled = sum_pooled / mask_sum 
        
        out, _ = self.gru(x_pooled)
        out = self.gru_norm(out[:, -1, :])
        return self.head(out)

def evaluate_only(W, cfg, val_df, device):
    print(f"[{time.strftime('%H:%M:%S')}] Evaluating saved model for W={W}...", flush=True)
    val_ds = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    edge_index = val_ds.edge_index
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = MaskedPhysicalGNN_GRU(
        node_dim=val_ds.F_node, edge_dim=val_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model_path = os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', f'exp_6K_W{W}_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    val_labels, val_probs = get_probs(model, val_loader, device, edge_index)

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

    attack_indices = np.where(val_labels == 1)[0]
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1) if len(attack_indices) > 0 else []
    
    det_count = 0
    event_metrics = []
    for i, event in enumerate(events):
        ev_probs = val_probs[event]
        detected = np.any(ev_probs >= optimal_thr)
        delay = int(np.argmax(ev_probs >= optimal_thr)) if detected else -1
        if detected: det_count += 1
        event_metrics.append({
            "event_id": i + 1,
            "duration": len(event),
            "detected": bool(detected),
            "delay": delay
        })

    return {
        "W": W,
        "Precision": float(val_prec),
        "Recall": float(val_rec),
        "F1": float(val_f1),
        "PR_AUC": float(val_pr),
        "ROC_AUC": float(val_roc),
        "Threshold": float(optimal_thr),
        "Event_Detection_Rate": f"{det_count}/{len(events)}",
        "Events": event_metrics
    }

def train_and_eval_w48(cfg, train_df, val_df, device):
    set_seed(42)
    W = 48
    B = 32  # Back to 32 for fast GPU utilization
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting Training W=48 with batch size {B} on {device}", flush=True)
    
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    val_ds   = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    
    edge_index = train_ds.edge_index
    e_idx_device = edge_index.to(device)

    labels_all = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    wts = np.where(labels_all == 1, 1.0 / n_a, 1.0 / n_n)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.float64), len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
    
    model = MaskedPhysicalGNN_GRU(
        node_dim=train_ds.F_node, edge_dim=train_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

    best_val_auc = 0.0
    patience, stall = 3, 0  # VERY fast early stopping for W=48 just to get a good result quickly without hanging
    save_path = os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', f'exp_6K_W{W}_model.pth')

    for ep in range(1, 10):
        model.train()
        t0 = time.time()
        for i, (xb_n, xb_e, _, yb) in enumerate(train_loader):
            xb_n, xb_e, yb = xb_n.to(device), xb_e.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb_n, xb_e, e_idx_device).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        print(f"[{time.strftime('%H:%M:%S')}] Epoch {ep} train time: {time.time()-t0:.2f}s", flush=True)

        val_labels, val_probs = get_probs(model, val_loader, device, edge_index)
        val_auc = roc_auc_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            stall = 0
            torch.save(model.state_dict(), save_path)
            print(f"[{time.strftime('%H:%M:%S')}] [W={W}] Epoch {ep}: Improved AUC to {val_auc:.4f}", flush=True)
        else:
            stall += 1
            print(f"[{time.strftime('%H:%M:%S')}] [W={W}] Epoch {ep}: No improvement ({stall}/{patience})", flush=True)
            if stall >= patience:
                break

    return evaluate_only(W, cfg, val_df, device)

def main():
    device = torch.device('cuda')
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))

    results = []
    for W in [12, 24, 36]:
        res = evaluate_only(W, cfg, val_df, device)
        results.append(res)
        
    res_48 = train_and_eval_w48(cfg, train_df, val_df, device)
    results.append(res_48)
        
    out_path = os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'phase_6K_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    md = "# Phase 6K: Temporal Window Sweep (E1 Masked Mean)\n\n"
    md += "| Window (W) | PR-AUC | ROC-AUC | F1 Score | Precision | Recall | Threshold | Event Detection |\n"
    md += "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        md += f"| {r['W']} | {r['PR_AUC']:.4f} | {r['ROC_AUC']:.4f} | {r['F1']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['Threshold']:.2f} | {r['Event_Detection_Rate']} |\n"
        
    md += "\n## Event-Level Breakdown\n"
    for r in results:
        md += f"\n### Window = {r['W']}\n"
        for ev in r['Events']:
            md += f"- Event {ev['event_id']} ({ev['duration']} frames): Detected={ev['detected']}, Delay={ev['delay']}\n"
            
    md_path = os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'phase_6K_sweep_results.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
        
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)

if __name__ == '__main__':
    main()
