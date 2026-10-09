import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.physical_dataset import WDSPhysicalGraphDataset
from models.gnn_gru.train_gnn_gru_physical import PhysicalGNN_GRU, get_probs, point_adjust

def evaluate_real_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(base_dir, 'data', 'processed')
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_processed.csv'))
    
    W = 12
    test_ds  = WDSPhysicalGraphDataset(test_df, base_dir, window_size=W, use_physical_attributes=False)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    edge_index = test_ds.edge_index

    model = PhysicalGNN_GRU(
        node_dim=test_ds.F_node, edge_dim=test_ds.F_edge, 
        gnn_hidden=64, gru_hidden=128
    ).to(device)

    # Use the headline model (without attrs)
    save_path = os.path.join(base_dir, 'models', 'gnn_gru', 'physical_gnn_gru_model.pth')
    if not os.path.exists(save_path):
        save_path = os.path.join(base_dir, 'models', 'gnn_gru', 'physical_gnn_gru_model.pth')
        
    print(f"Loading {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_labels, test_probs = get_probs(model, test_loader, device, edge_index)
    
    best_pw_f1 = 0
    best_pw_thr = 0
    best_pa_f1 = 0
    for t in np.arange(0.01, 0.99, 0.01):
        preds = (test_probs >= t).astype(int)
        
        # Pointwise
        pw_f1 = f1_score(test_labels, preds, zero_division=0)
        if pw_f1 > best_pw_f1:
            best_pw_f1 = pw_f1
            best_pw_thr = t
            
        # PA
        pa_preds = point_adjust(test_labels, preds)
        pa_f1 = f1_score(test_labels, pa_preds, zero_division=0)
        if pa_f1 > best_pa_f1:
            best_pa_f1 = pa_f1

    print(f"\nMaximum Pointwise F1: {best_pw_f1:.4f} at Threshold: {best_pw_thr:.2f}")
    print(f"Maximum PA F1:        {best_pa_f1:.4f}")

if __name__ == "__main__":
    evaluate_real_test()
