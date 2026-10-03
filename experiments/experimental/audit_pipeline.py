import os
import pandas as pd
import numpy as np
import json
import torch
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU

def compute_stats(tensor):
    return {
        "min": float(np.min(tensor)),
        "max": float(np.max(tensor)),
        "mean": float(np.mean(tensor)),
        "std": float(np.std(tensor)),
        "zeros_pct": float(np.mean(tensor == 0.0) * 100)
    }

def audit():
    print("Starting Physical Pipeline Audit...")
    
    # Load data
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'train_processed.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'val_processed.csv'))
    
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=12, bidirectional=False)
    val_ds = WDSPhysicalGraphDataset(val_df, base_dir, window_size=12, bidirectional=False)
    
    # 1. Inspect Node Features
    node_names = ['measurement', 'observation_mask', 'elevation', 'has_elevation']
    edge_names = ['flow', 'status', 'observation_mask', 'length', 'diameter', 'roughness', 
                  'has_len', 'has_diam', 'has_rough', 'is_pipe', 'is_pump', 'is_valve']
    
    audit_results = {
        "node_features": {},
        "edge_features": {},
        "train_val_shift": {},
        "model_architecture": {}
    }
    
    for i, name in enumerate(node_names):
        train_feat = train_ds.node_tensor[:, :, i]
        val_feat = val_ds.node_tensor[:, :, i]
        audit_results["node_features"][name] = {
            "train": compute_stats(train_feat),
            "val": compute_stats(val_feat)
        }
        
    for i, name in enumerate(edge_names):
        train_feat = train_ds.edge_tensor[:, :, i]
        val_feat = val_ds.edge_tensor[:, :, i]
        audit_results["edge_features"][name] = {
            "train": compute_stats(train_feat),
            "val": compute_stats(val_feat)
        }
        
    # Check graph sparsity
    mask = train_ds.node_tensor[0, :, 1]
    n_nodes = len(mask)
    n_observed = int(np.sum(mask))
    
    audit_results["model_architecture"]["graph_sparsity"] = {
        "total_nodes": n_nodes,
        "observed_nodes": n_observed,
        "unobserved_nodes": n_nodes - n_observed
    }
    
    with open(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'phase_6C_physical_audit.json'), 'w') as f:
        json.dump(audit_results, f, indent=4)
        
    print("Audit JSON saved. Generating MD...")
    
    md_content = "# Phase 6C: Physical Pipeline Audit Report\n\n"
    md_content += "## 1. Feature Scales (Magnitude Mismatch)\n"
    for name, stats in audit_results["node_features"].items():
        md_content += f"- **Node: {name}**: Mean={stats['train']['mean']:.3f}, Std={stats['train']['std']:.3f} (Max={stats['train']['max']:.3f})\n"
    for name, stats in audit_results["edge_features"].items():
        md_content += f"- **Edge: {name}**: Mean={stats['train']['mean']:.3f}, Std={stats['train']['std']:.3f} (Max={stats['train']['max']:.3f})\n"
        
    with open(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'phase_6C_physical_audit.md'), 'w') as f:
        f.write(md_content)
        
if __name__ == "__main__":
    audit()
