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

from models.gnn_gru.sensor_dataset import WDSSensorGraphDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# B13: Pure PyTorch implementation of a Physical Message Passing Layer
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
        
        B_W, E, _ = messages.shape
        N = x.shape[1]
        
        agg_messages = torch.zeros((B_W, N, messages.shape[-1]), device=x.device)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B_W, E, messages.shape[-1])
        agg_messages.scatter_add_(1, dst_expanded, messages)
        
        node_inputs = torch.cat([x, agg_messages], dim=-1)
        x_out = self.node_mlp(node_inputs)
        
        if x.shape[-1] == x_out.shape[-1]:
            x_out = x_out + x
            
        return self.norm(x_out)

# B14: Integrate Physical GNN with Temporal GRU
class PhysicalGNN_GRU(nn.Module):
    def __init__(self, node_dim=3, edge_dim=9, gnn_hidden=64, gru_hidden=128):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, gnn_hidden)
        
        self.gnn1 = PhysicalGNNLayer(gnn_hidden, edge_dim, gnn_hidden)
        self.gnn2 = PhysicalGNNLayer(gnn_hidden, edge_dim, gnn_hidden)
        
        self.gru = nn.GRU(gnn_hidden, gru_hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_attr, edge_index):
        B, W, N, F_n = x.shape
        _, _, E, F_e = edge_attr.shape
        
        x_flat = x.view(B * W, N, F_n)
        e_flat = edge_attr.view(B * W, E, F_e)
        
        x_h = self.node_proj(x_flat)
        x_h = self.gnn1(x_h, e_flat, edge_index)
        x_h = self.gnn2(x_h, e_flat, edge_index)
        
        # Readout: Mean over all 396 physical nodes
        z_graph = x_h.mean(dim=1)
        z_graph = z_graph.view(B, W, -1)
        
        gru_out, _ = self.gru(z_graph)
        gru_last = self.gru_norm(gru_out[:, -1, :])
        
        return self.head(gru_last)

def get_probs(model, loader, device, edge_index, log_stats=False):
    model.eval()
    all_labels, all_probs, all_logits = [], [], []
    edge_index = edge_index.to(device)
    with torch.no_grad():
        for xb_n, xb_e, _, yb in loader:
            xb_n, xb_e = xb_n.to(device), xb_e.to(device)
            out = model(xb_n, xb_e, edge_index).squeeze(-1)
            logits = out.cpu().numpy().tolist()
            probs = torch.sigmoid(out).cpu().numpy().tolist()
            if isinstance(probs, float): 
                probs = [probs]
                logits = [logits]
            all_probs.extend(probs)
            all_logits.extend(logits)
            all_labels.extend(yb.numpy().tolist())
            
    all_labels = np.array(all_labels, dtype=int)
    all_probs = np.array(all_probs, dtype=float)
    all_logits = np.array(all_logits, dtype=float)
    
    if log_stats:
        norm_mask = (all_labels == 0)
        atk_mask = (all_labels == 1)
        
        print("\n--- Validation Probability/Logit Diagnostics (Steps 3 & 4) ---")
        if norm_mask.sum() > 0:
            norm_probs = all_probs[norm_mask]
            norm_logits = all_logits[norm_mask]
            print(f"Normal | Probs - Mean: {norm_probs.mean():.4f}, Std: {norm_probs.std():.4f}, Min: {norm_probs.min():.4f}, Max: {norm_probs.max():.4f}")
            print(f"Normal | Logit - Mean: {norm_logits.mean():.4f}, Std: {norm_logits.std():.4f}")
        if atk_mask.sum() > 0:
            atk_probs = all_probs[atk_mask]
            atk_logits = all_logits[atk_mask]
            print(f"Attack | Probs - Mean: {atk_probs.mean():.4f}, Std: {atk_probs.std():.4f}, Min: {atk_probs.min():.4f}, Max: {atk_probs.max():.4f}")
            print(f"Attack | Logit - Mean: {atk_logits.mean():.4f}, Std: {atk_logits.std():.4f}")
            
    return all_labels, all_probs

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

def train_sensor_gnn(use_physical_attributes=True, model_name="sensor_gnn_gru_model.pth"):
    set_seed(42)
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'models', 'gnn_gru')
    docs_dir = os.path.join(base_dir, 'docs')
    save_path = os.path.join(output_dir, model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")
    print(f"Ablation Mode: use_physical_attributes = {use_physical_attributes}")

    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))

    W, B = 12, 128
    
    print("Building spatio-temporal WDS datasets...")
    train_ds = WDSSensorGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=use_physical_attributes)
    val_ds   = WDSSensorGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=use_physical_attributes)
    test_ds  = WDSSensorGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=use_physical_attributes)
    
    edge_index = train_ds.edge_index

    labels_all = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    n_n = (labels_all == 0).sum()
    n_a = (labels_all == 1).sum()
    wts = np.where(labels_all == 1, 1.0 / n_a, 1.0 / n_n)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.float64), len(train_ds), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=B, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=B, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=B, shuffle=False)
    
    node_dim = train_ds.F_node # 4
    edge_dim = train_ds.F_edge # 12
    
    model = PhysicalGNN_GRU(node_dim=node_dim, edge_dim=edge_dim, gnn_hidden=64, gru_hidden=128).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_val_auc = 0.0
    patience, stall = 15, 0
    epochs = 60
    e_idx_device = edge_index.to(device)

    print("Starting Sensor GNN-GRU Training...")
    for ep in range(1, epochs + 1):
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
        scheduler.step()

        val_labels, val_probs = get_probs(model, val_loader, device, edge_index, log_stats=False)
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
    # Step 3 & 4: Log stats on the best validation checkpoint
    val_labels, val_probs   = get_probs(model, val_loader, device, edge_index, log_stats=True)
    test_labels, test_probs = get_probs(model, test_loader, device, edge_index, log_stats=False)

    best_f1, optimal_thr = 0, 0.5
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (val_probs >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            optimal_thr = t

    print(f"\nSelected Validation Threshold (PA F1): {optimal_thr:.4f} (Val F1: {best_f1:.4f})")

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
        "Ablation_Physical_Attrs": use_physical_attributes,
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
    
    metrics_file = f'sensor_gnn_evaluation_{"with" if use_physical_attributes else "without"}_attrs.json'
    with open(os.path.join(docs_dir, metrics_file), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved evaluation metrics -> docs/{metrics_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-attrs', action='store_true', help='Run Model C: without physical attributes')
    args = parser.parse_args()
    
    if args.no_attrs:
        train_sensor_gnn(use_physical_attributes=False, model_name="sensor_gnn_gru_model_no_attrs.pth")
    else:
        train_sensor_gnn(use_physical_attributes=True, model_name="sensor_gnn_gru_model_E.pth")
