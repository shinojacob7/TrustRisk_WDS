import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shap
import sys
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.tune_gnn_gru import ConfigurablePhysicalGNN_GRU

class GNNWrapper(nn.Module):
    def __init__(self, base_model, edge_index, temperature=1.0):
        super().__init__()
        self.base_model = base_model
        self.edge_index = edge_index
        self.temperature = temperature
        
    def forward(self, xb_n, xb_e):
        logits = self.base_model(xb_n, xb_e, self.edge_index)
        return torch.sigmoid(logits / self.temperature)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def gini_coefficient(x):
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x)) if np.mean(x) > 0 else 0.0

def run_trust_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Trust Pipeline on {device}")
    
    with open(os.path.join(base_dir, 'experiments', 'tuning', 'best_config.json'), 'r') as f:
        cfg = json.load(f)
    with open(os.path.join(base_dir, 'experiments', 'final', 'temperature.json'), 'r') as f:
        t_cfg = json.load(f)
    opt_T = t_cfg['Optimal_Temperature']
    
    data_dir = os.path.join(base_dir, 'data', 'processed')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_processed.csv'))
    val_df   = pd.read_csv(os.path.join(data_dir, 'val_processed.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    W, B = 12, 128
    train_ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=W, use_physical_attributes=True)
    val_ds   = WDSPhysicalGraphDataset(val_df, base_dir, window_size=W, use_physical_attributes=True)
    test_ds  = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=True)
    
    val_loader  = DataLoader(val_ds, batch_size=B, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=B, shuffle=False)
    
    model = ConfigurablePhysicalGNN_GRU(
        node_dim=test_ds.F_node, edge_dim=test_ds.F_edge, 
        gnn_hidden=cfg['gnn_hidden'], gru_hidden=cfg['gru_hidden'],
        num_gnn_layers=cfg['num_gnn_layers'], num_gru_layers=cfg['num_gru_layers'],
        dropout=cfg['dropout']
    ).to(device)
    
    model_path = os.path.join(base_dir, 'experiments', 'final', 'final_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    edge_index = test_ds.edge_index.to(device)
    wrapper = GNNWrapper(model, edge_index, opt_T).to(device)
    
    # ---------------------------------------------------------
    # PHASE A: U_min and U_max strictly from Validation Data
    # ---------------------------------------------------------
    print("Phase A: Establishing U_min and U_max from Validation Set...")
    model.eval()
    enable_dropout(model)
    
    val_passes = []
    with torch.no_grad():
        for pass_idx in range(30): # 30 passes is sufficient for variance bounds
            pass_probs = []
            for xb_n, xb_e, _, _ in val_loader:
                cal_probs = wrapper(xb_n.to(device), xb_e.to(device)).squeeze(-1)
                pass_probs.append(cal_probs.cpu())
            val_passes.append(torch.cat(pass_probs).numpy())
            
    val_passes = np.array(val_passes)
    val_variances = np.var(val_passes, axis=0)
    u_min, u_max = np.min(val_variances), np.max(val_variances)
    print(f"Validation Bounds -> U_min: {u_min:.6f}, U_max: {u_max:.6f}")
    
    # ---------------------------------------------------------
    # PHASE B: Test Set Evaluation (p_attack, C, U_norm, ECP)
    # ---------------------------------------------------------
    print("Phase B: Processing Test Set (MC Dropout)...")
    test_passes = []
    with torch.no_grad():
        for pass_idx in range(50):
            pass_probs = []
            for xb_n, xb_e, _, _ in test_loader:
                cal_probs = wrapper(xb_n.to(device), xb_e.to(device)).squeeze(-1)
                pass_probs.append(cal_probs.cpu())
            test_passes.append(torch.cat(pass_probs).numpy())
            
    test_passes = np.array(test_passes)
    p_attack = np.mean(test_passes, axis=0)
    u_raw = np.var(test_passes, axis=0)
    
    # Calculate C and U_norm
    C = np.maximum(p_attack, 1.0 - p_attack)
    u_norm = np.clip((u_raw - u_min) / (u_max - u_min + 1e-9), 0.0, 1.0)
    
    print("Phase B: Processing Test Set (SHAP ECP)...")
    # Small background for speed
    bg_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    bg_n, bg_e, _, _ = next(iter(bg_loader))
    explainer = shap.GradientExplainer(wrapper, [bg_n.to(device), bg_e.to(device)])
    
    torch.backends.cudnn.enabled = False
    
    ecp_scores = []
    test_loader_shap = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    for xb_n, xb_e, _, _ in test_loader_shap:
        shap_vals = explainer.shap_values([xb_n.to(device), xb_e.to(device)])
        shap_n, shap_e = shap_vals[0], shap_vals[1]
        
        for i in range(xb_n.shape[0]):
            node_imp = np.abs(shap_n[i]).sum(axis=(0, 2))
            edge_imp = np.abs(shap_e[i]).sum(axis=(0, 2))
            spatial_imp = np.concatenate([node_imp, edge_imp])
            ecp_scores.append(gini_coefficient(spatial_imp))
            
    torch.backends.cudnn.enabled = True
    ecp = np.array(ecp_scores)
    
    # ---------------------------------------------------------
    # PHASE C: Integration & Storage
    # ---------------------------------------------------------
    print("Phase C: Calculating Trust Score and Saving Pipeline...")
    alpha = beta = gamma = 1.0
    TS = (C ** alpha) * ((1.0 - u_norm) ** beta) * (ecp ** gamma)
    
    labels = np.array([test_ds[i][3].item() for i in range(len(test_ds))])
    
    df_trust = pd.DataFrame({
        'frame_idx': np.arange(len(test_ds)),
        'true_label': labels,
        'p_attack': p_attack,
        'C': C,
        'U_raw': u_raw,
        'U_norm': u_norm,
        'ECP': ecp,
        'Trust_Score': TS
    })
    
    out_path = os.path.join(base_dir, 'experiments', 'final', 'trust_pipeline_virgin.csv')
    df_trust.to_csv(out_path, index=False)
    
    print(f"Saved complete auditable Trust pipeline to {out_path}")
    print("Sample Output:")
    print(df_trust.head(10).to_string())
    
if __name__ == '__main__':
    run_trust_pipeline()
