import os
import json
import pandas as pd
import numpy as np
import torch
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import point_adjust, get_probs, PhysicalGNNLayer

# Load E1 Model class definition to load weights safely
class MaskedPhysicalGNN_GRU(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, gnn_hidden, gru_hidden, num_gnn_layers, num_gru_layers, dropout):
        super().__init__()
        self.node_proj = torch.nn.Linear(node_dim, gnn_hidden)
        self.gnn_layers = torch.nn.ModuleList([
            PhysicalGNNLayer(gnn_hidden, edge_dim, gnn_hidden) for _ in range(num_gnn_layers)
        ])
        self.gru = torch.nn.GRU(gnn_hidden, gru_hidden, num_layers=num_gru_layers, batch_first=True, dropout=dropout if num_gru_layers > 1 else 0)
        self.gru_norm = torch.nn.LayerNorm(gru_hidden)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(gru_hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def extract_node_embeddings(self, x, edge_attr, edge_index):
        B, W, N, F = x.shape
        x_flat = x.view(B * W, N, F)
        e_flat = edge_attr.view(B * W, edge_attr.shape[2], edge_attr.shape[3])
        
        x_gnn = self.node_proj(x_flat)
        for gnn in self.gnn_layers:
            x_gnn = gnn(x_gnn, e_flat, edge_index)
            
        x_gnn = x_gnn.view(B, W, N, -1)
        return x_gnn

def main():
    val_raw = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'val.csv'))
    val_proc = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'val_processed.csv'))
    
    # Extract labels and events
    labels = val_raw['ATT_FLAG'].values
    attack_indices = np.where(labels == 1)[0]
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1) if len(attack_indices) > 0 else []
    
    # 1. Map observed nodes
    obs_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'ctown_node_observation_mask.csv'))
    map_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'batadal_epanet_mapping.csv'))
    
    observed_nodes = obs_df[obs_df['observed'] == 1].copy()
    mapped = pd.merge(observed_nodes, map_df, left_on='node_id', right_on='element', how='left')
    mapped = mapped[mapped['graph_role'] == 'node'].dropna(subset=['variable'])
    
    node_idx_to_var = dict(zip(mapped['node_index'], mapped['variable']))
    observed_indices = sorted(list(node_idx_to_var.keys()))
    observed_vars = [node_idx_to_var[i] for i in observed_indices]
    
    print(f"Mapped {len(observed_indices)} observed nodes to variables.")
    
    # 2. Event analysis
    event_stats = {}
    for ev_id, event in enumerate(events):
        start = event[0]
        end = event[-1]
        
        prev_end = 0 if ev_id == 0 else events[ev_id-1][-1]
        baseline_start = prev_end + 1
        baseline_end = start - 1
        
        if baseline_start >= baseline_end:
            baseline_start = max(0, start - 100) # fallback
            
        baseline_df = val_raw.iloc[baseline_start:baseline_end + 1]
        event_df = val_raw.iloc[start:end + 1]
        
        sensors_stats = {}
        max_abs_z_all = 0
        z_ge_2 = 0
        z_ge_3 = 0
        z_ge_5 = 0
        
        for var in observed_vars:
            mu = baseline_df[var].mean()
            sigma = baseline_df[var].std()
            if sigma == 0 or pd.isna(sigma): sigma = 1.0 # avoid div by zero
            
            ev_min = event_df[var].min()
            ev_max = event_df[var].max()
            ev_mean = event_df[var].mean()
            
            z_scores = (event_df[var] - mu) / sigma
            abs_z = np.abs(z_scores)
            max_abs_z = abs_z.max()
            peak_time = start + abs_z.idxmax() - start
            
            sensors_stats[var] = {
                "mu": float(mu),
                "sigma": float(sigma),
                "min": float(ev_min),
                "max": float(ev_max),
                "mean": float(ev_mean),
                "max_abs_z": float(max_abs_z),
                "detectable_z3": bool(max_abs_z >= 3)
            }
            
            max_abs_z_all = max(max_abs_z_all, max_abs_z)
            if max_abs_z >= 2: z_ge_2 += 1
            if max_abs_z >= 3: z_ge_3 += 1
            if max_abs_z >= 5: z_ge_5 += 1
            
        event_stats[f"Event_{ev_id+1}"] = {
            "start": int(start),
            "end": int(end),
            "duration": int(len(event)),
            "sensors": sensors_stats,
            "z_ge_2": z_ge_2,
            "z_ge_3": z_ge_3,
            "z_ge_5": z_ge_5,
            "max_z_all": float(max_abs_z_all)
        }
        
    # 3. GNN Embedding Analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
        
    val_ds = WDSPhysicalGraphDataset(val_proc, base_dir, window_size=12, use_physical_attributes=True)
    e_idx_device = val_ds.edge_index.to(device)
    
    model = MaskedPhysicalGNN_GRU(
        node_dim=val_ds.F_node, edge_dim=val_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'exp_E1_masked_mean_model.pth')))
    model.eval()
    
    gnn_stats = {}
    with torch.no_grad():
        for ev_id, event in enumerate(events):
            # Evaluate embeddings for the event frames
            # dataset index = frame - window_size + 1
            idxs = [f - 11 for f in event if f - 11 >= 0]
            if not idxs: continue
            
            X_w = torch.stack([val_ds[i][0] for i in idxs]).to(device)
            E_w = torch.stack([val_ds[i][1] for i in idxs]).to(device)
            
            x_gnn = model.extract_node_embeddings(X_w, E_w, e_idx_device)
            # x_gnn shape: [B, W, N, hidden]
            # we only care about the last frame of the window (W-1)
            x_gnn_last = x_gnn[:, -1, :, :] # [B, N, hidden]
            
            # extract the 19 observed nodes
            x_obs = x_gnn_last[:, observed_indices, :] # [B, 19, hidden]
            
            # calculate magnitude (L2 norm) of embeddings
            mags = torch.norm(x_obs, dim=2) # [B, 19]
            
            mean_mag = mags.mean().item()
            max_mag = mags.max().item()
            std_mag = mags.std().item()
            
            gnn_stats[f"Event_{ev_id+1}"] = {
                "mean_mag": mean_mag,
                "max_mag": max_mag,
                "std_mag": std_mag
            }
            
    # Combine everything
    out = {
        "event_3_interval": {
            "start": event_stats["Event_3"]["start"],
            "end": event_stats["Event_3"]["end"],
            "duration": event_stats["Event_3"]["duration"]
        },
        "observed_node_mapping": node_idx_to_var,
        "event_stats": event_stats,
        "gnn_embeddings": gnn_stats
    }
    
    with open(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'phase_6I_event3_analysis.json'), 'w') as f:
        json.dump(out, f, indent=4)
        
    print("Done")

if __name__ == '__main__':
    main()
