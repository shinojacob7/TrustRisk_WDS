import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset

# Re-implement get_probs, point_adjust, get_event_metrics for validation
def point_adjust(labels, preds):
    pa_preds = preds.copy()
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        return pa_preds
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    for event in events:
        if np.any(preds[event] == 1):
            pa_preds[event] = 1
    return pa_preds

def get_event_metrics(labels, preds):
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        fp_mask = (preds == 1) & (labels == 0)
        fp_indices = np.where(fp_mask)[0]
        false_alarms = 0
        if len(fp_indices) > 0:
            fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
            false_alarms = len(fp_events)
        return 0, false_alarms
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    detected_events = 0
    for event in events:
        if np.any(preds[event] == 1):
            detected_events += 1
    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = np.where(fp_mask)[0]
    false_alarms = 0
    if len(fp_indices) > 0:
        fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
        false_alarms = len(fp_events)
    return detected_events, false_alarms

def get_probs(model, loader, device, edge_index):
    model.eval()
    all_probs = []
    all_labels = []
    e_idx_device = edge_index.to(device)
    with torch.no_grad():
        for xb_n, xb_e, _, yb in loader:
            xb_n, xb_e, yb = xb_n.to(device), xb_e.to(device), yb.to(device)
            out = model(xb_n, xb_e, e_idx_device).squeeze(-1)
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

class PhysicalGNNLayer(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_in_dim * 2 + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_attr, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x_src = x[:, src, :]
        x_dst = x[:, dst, :]
        
        edge_inputs = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        messages = self.edge_mlp(edge_inputs)
        
        B_W = x.shape[0]
        E = edge_index.shape[1]
        N = x.shape[1]
        
        agg_messages = torch.zeros((B_W, N, messages.shape[-1]), device=x.device)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B_W, E, messages.shape[-1])
        agg_messages.scatter_add_(1, dst_expanded, messages)
        
        node_inputs = torch.cat([x, agg_messages], dim=-1)
        x_out = self.node_mlp(node_inputs)
        
        if x.shape[-1] == x_out.shape[-1]:
            x_out = x_out + x
            
        return self.norm(x_out)

class ConfigurablePhysicalGNN_GRU(nn.Module):
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
        # pool over nodes
        x_pooled = x_gnn.mean(dim=2) 
        
        out, _ = self.gru(x_pooled)
        out = self.gru_norm(out[:, -1, :])
        return self.head(out)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_tuning():
    data_dir = os.path.join(base_dir, 'data', 'processed')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Tuning on {device}")
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    
    import gc
    # We will fix window size = 12 for all to save time, as 12 is optimal standard.
    W = 12
    B = 64  # Reduced from 128 to 64 to prevent CUDA OOM
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    val_ds   = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    
    edge_index = train_ds.edge_index
    e_idx_device = edge_index.to(device)

    labels_all = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    wts = np.where(labels_all == 1, 1.0 / n_a, 1.0 / n_n)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.float64), len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=B, shuffle=False)
    
    node_dim = train_ds.F_node
    edge_dim = train_ds.F_edge

    # Define hyperparameter grid
    configs = [
        {'gnn_hidden': 32, 'gru_hidden': 64,  'num_gnn_layers': 2, 'num_gru_layers': 1, 'lr': 1e-3, 'dropout': 0.1, 'wd': 1e-4},
        {'gnn_hidden': 64, 'gru_hidden': 128, 'num_gnn_layers': 2, 'num_gru_layers': 2, 'lr': 3e-4, 'dropout': 0.2, 'wd': 1e-4}, # baseline D
        {'gnn_hidden': 64, 'gru_hidden': 128, 'num_gnn_layers': 3, 'num_gru_layers': 2, 'lr': 1e-4, 'dropout': 0.3, 'wd': 1e-5},
        {'gnn_hidden': 64, 'gru_hidden': 64,  'num_gnn_layers': 2, 'num_gru_layers': 2, 'lr': 1e-3, 'dropout': 0.2, 'wd': 1e-5},
        {'gnn_hidden': 128,'gru_hidden': 128, 'num_gnn_layers': 2, 'num_gru_layers': 1, 'lr': 3e-4, 'dropout': 0.1, 'wd': 1e-4}
    ]
    
    results = []
    
    for i, cfg in enumerate(configs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        set_seed(42)
        print(f"\\n--- Running Config {i+1}/{len(configs)} ---")
        print(cfg)
        
        model = ConfigurablePhysicalGNN_GRU(
            node_dim=node_dim, edge_dim=edge_dim, 
            gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
            num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
            dropout=cfg['dropout']
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        
        # Train for max 25 epochs (since it's a sweep)
        best_val_auc = 0.0
        best_state = None
        stall = 0
        
        for ep in range(1, 26):
            model.train()
            train_loss = 0.0
            for xb_n, xb_e, _, yb in train_loader:
                xb_n, xb_e, yb = xb_n.to(device), xb_e.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb_n, xb_e, e_idx_device).squeeze(-1)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                
            val_labels, val_probs = get_probs(model, val_loader, device, edge_index)
            val_auc = roc_auc_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = model.state_dict().copy()
                stall = 0
            else:
                stall += 1
                if stall >= 7:
                    break
                    
        # Evaluate on Validation Set using best weights
        model.load_state_dict(best_state)
        val_labels, val_probs = get_probs(model, val_loader, device, edge_index)
        
        # Find best threshold on Val
        best_f1 = 0
        opt_t = 0.5
        for t in np.arange(0.01, 0.99, 0.01):
            preds = (val_probs >= t).astype(int)
            pa_preds = point_adjust(val_labels, preds)
            f1 = f1_score(val_labels, pa_preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                opt_t = t
                
        # Get Event Metrics on Val
        val_preds = (val_probs >= opt_t).astype(int)
        val_det, val_fa = get_event_metrics(val_labels, val_preds)
        
        print(f"Val AUC: {best_val_auc:.4f} | Val PA-F1: {best_f1:.4f} | Det: {val_det}/7 | FA: {val_fa}")
        
        cfg_result = cfg.copy()
        cfg_result['Val_AUC'] = best_val_auc
        cfg_result['Val_PA_F1'] = best_f1
        cfg_result['Val_Detected'] = val_det
        cfg_result['Val_FA'] = val_fa
        cfg_result['Val_Threshold'] = opt_t
        results.append(cfg_result)
        
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.join(base_dir, 'experiments', 'tuning'), exist_ok=True)
    df_results.to_csv(os.path.join(base_dir, 'experiments', 'tuning', 'model_d_tuning_results.csv'), index=False)
    
    # Select best config
    # Precedence: Highest Detected, then Lowest FA, then Highest PA-F1
    best_config = df_results.sort_values(by=['Val_Detected', 'Val_FA', 'Val_PA_F1'], ascending=[False, True, False]).iloc[0]
    best_config_dict = best_config.to_dict()
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'w') as f:
        json.dump(best_config_dict, f, indent=4)
        
    print("\n=== Tuning Complete ===")
    print(f"Best Config saved to experiments/tuning/best_config.json")
    print(best_config)

if __name__ == '__main__':
    run_tuning()
