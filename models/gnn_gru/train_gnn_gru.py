import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class WDSGraphDataset(Dataset):
    def __init__(self, df, window_size=12, target_col='ATT_FLAG'):
        self.window_size = window_size
        feat_cols = [c for c in df.columns if c not in (target_col, 'DATETIME')]
        self.features = df[feat_cols].values.astype(np.float32)
        self.labels   = df[target_col].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.features) - self.window_size + 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx : idx + self.window_size])
        y = torch.tensor(self.labels[idx + self.window_size - 1])
        return x, y

class GNNLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_f, out_f))
        self.norm = nn.LayerNorm(out_f)
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, adj):
        return self.norm(F.relu(torch.matmul(adj, torch.matmul(x, self.W))))

class ZoneAdapter(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        b = max(1, dim // r)
        self.net = nn.Sequential(nn.Linear(dim, b), nn.GELU(), nn.Linear(b, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))

class GNN_GRU(nn.Module):
    def __init__(self, num_nodes, adj_matrix, zone_mapping, num_zones,
                 gnn_hidden=48, gru_hidden=128, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.zone_mapping = zone_mapping

        self.register_buffer('adj', torch.tensor(adj_matrix, dtype=torch.float32))
        self.gnn1 = GNNLayer(1, gnn_hidden)
        self.gnn2 = GNNLayer(gnn_hidden, gnn_hidden)
        self.gru = nn.GRU(gnn_hidden, gru_hidden, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.adapters = nn.ModuleList([ZoneAdapter(gnn_hidden) for _ in range(num_zones)])
        self.node_attn = nn.Linear(gnn_hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + gnn_hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, S, N = x.size()
        xr = x.view(B * S, N, 1)
        z = self.gnn2(self.gnn1(xr, self.adj), self.adj)
        z = z.view(B, S, N, -1)
        h_g = self.gru_norm(self.gru(z.mean(dim=2))[0][:, -1, :])
        z_last = z[:, -1, :, :]
        parts = [self.adapters[self.zone_mapping[n]](z_last[:, n, :]) for n in range(N)]
        z_last = torch.stack(parts, dim=1)
        w = torch.softmax(self.node_attn(z_last), dim=1)
        h_n = (w * z_last).sum(dim=1)
        return self.head(torch.cat([h_g, h_n], dim=1))

# Note: In Stage B, this will be replaced with the 396-node physical graph!
# Keeping this strictly derived from train data for now to fix leakage in Stage A.
def build_sensor_adjacency(df):
    feat = [c for c in df.columns if c not in ('DATETIME', 'ATT_FLAG')]
    corr = df[feat].corr().abs().fillna(0).values.copy()
    np.fill_diagonal(corr, 1.0)
    adj = np.where(corr > 0.3, corr, 0.0)
    import igraph as ig
    import leidenalg
    rows, cols = np.where(adj > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights = adj[adj > 0].tolist()
    g = ig.Graph(n=len(feat), edges=edges, directed=False)
    g.es['weight'] = weights
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=g.es['weight'])
    zone_map = part.membership
    n_zones = max(zone_map) + 1
    row_sum = adj.sum(axis=1, keepdims=True)
    norm_adj = adj / (row_sum + 1e-8)
    return norm_adj, len(feat), zone_map, n_zones

def get_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).squeeze(-1)
            probs = torch.sigmoid(out).cpu().numpy().tolist()
            if isinstance(probs, float): probs = [probs]
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().tolist())
    return np.array(all_labels, dtype=int), np.array(all_probs, dtype=float)

def point_adjust(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    adjusted_preds = preds.copy()
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0: return adjusted_preds
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    for event in events:
        if np.any(preds[event] == 1):
            adjusted_preds[event] = 1
    return adjusted_preds

def get_event_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    
    attack_indices = np.where(labels == 1)[0]
    
    if len(attack_indices) == 0:
        fp_mask = (preds == 1)
        fp_indices = np.where(fp_mask)[0]
        false_alarms = 0
        if len(fp_indices) > 0:
            fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
            false_alarms = len(fp_events)
        return 0.0, 0.0, 0, 0, false_alarms
        
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    
    detected_events = 0
    total_delay = 0
    
    for event in events:
        event_preds = preds[event]
        if np.any(event_preds == 1):
            detected_events += 1
            delay = np.argmax(event_preds == 1)
            total_delay += delay
            
    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = np.where(fp_mask)[0]
    false_alarms = 0
    if len(fp_indices) > 0:
        fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
        false_alarms = len(fp_events)
        
    edr = detected_events / len(events)
    mean_delay = total_delay / detected_events if detected_events > 0 else 0
    
    return edr, mean_delay, detected_events, len(events), false_alarms

def apply_threshold(labels, probs, thr):
    raw_preds = (probs >= thr).astype(int)
    pa_preds = point_adjust(labels, raw_preds)
    
    f1   = f1_score(labels, pa_preds, zero_division=0)
    prec = precision_score(labels, pa_preds, zero_division=0)
    rec  = recall_score(labels, pa_preds, zero_division=0)
    acc  = accuracy_score(labels, pa_preds)
    
    cm = confusion_matrix(labels, pa_preds)
    edr, mean_delay, det_ev, tot_ev, false_alarms = get_event_metrics(labels, raw_preds)
    
    return acc, f1, prec, rec, cm, edr, mean_delay, det_ev, tot_ev, false_alarms

def train_gnn_gru():
    set_seed(42)  # Stage A Fix: Reproducible seeds
    
    base_dir  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir  = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'models', 'gnn_gru')
    os.makedirs(output_dir, exist_ok=True)
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'gnn_gru_model.pth')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    # Stage A Fix: Removing val_df from training set (No Leakage)
    # Using only train_df to build adjacency and train model.
    adj_matrix, num_nodes, zone_mapping, num_zones = build_sensor_adjacency(train_df)

    W, B = 12, 256
    train_ds = WDSGraphDataset(train_df, window_size=W)
    val_ds   = WDSGraphDataset(val_df, window_size=W)
    test_ds  = WDSGraphDataset(test_df, window_size=W)

    labels_all = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    wts = np.where(labels_all == 1, 1.0 / n_a, 1.0 / n_n)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.float64), len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=B, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=B, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    model = GNN_GRU(num_nodes, adj_matrix, zone_mapping, num_zones).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    best_val_auc = 0.0
    patience, stall = 20, 0
    epochs = 100

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        val_labels, val_probs = get_probs(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs) if val_labels.sum() > 0 else 0.0

        print(f"Epoch {ep:03d} | Loss: {train_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            stall = 0
            torch.save(model.state_dict(), save_path)
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stopping triggered at epoch {ep}")
                break

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(save_path))
    val_labels, val_probs   = get_probs(model, val_loader, device)
    test_labels, test_probs = get_probs(model, test_loader, device)

    # Stage A Fix: Threshold chosen exclusively on Validation Data
    best_f1, optimal_thr = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            optimal_thr = t

    print(f"Selected Validation Threshold (PA F1): {optimal_thr:.4f} (Val F1: {best_f1:.4f})")

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
    with open(os.path.join(docs_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved evaluation metrics -> docs/evaluation_metrics.json")

if __name__ == '__main__':
    train_gnn_gru()
