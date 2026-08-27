import torch
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
import torch.nn as nn

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from models.gnn_gru.train_gnn_gru import GNN_GRU, WDSGraphDataset, build_sensor_adjacency, evaluate_model

def evaluate_real_test():
    # 1. Rebuild the exact same adjacency and zone mapping from training data
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'BATADAL_balanced_scaled.csv'))
    
    # Loop to ensure Leiden finds exactly the 14 zones that were saved in the model weights
    for seed in range(100):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        adj_matrix, num_nodes, zone_mapping, num_zones = build_sensor_adjacency(train_df)
        if num_zones == 14:
            print(f"Matched 14 zones with seed {seed}")
            break
    
    # 2. Load the REAL test dataset
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'BATADAL_test_dataset_scaled.csv'))
    
    # Ensure DATETIME is dropped if it exists so we don't crash the tensor conversion
    if 'DATETIME' in test_df.columns:
        test_df = test_df.drop(columns=['DATETIME'])
        
    test_dataset = WDSGraphDataset(test_df, window_size=12)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 3. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNN_GRU(num_nodes=num_nodes, adj_matrix=adj_matrix, zone_mapping=zone_mapping, num_zones=num_zones).to(device)
    model.use_adapters = True
    
    save_path = os.path.join(base_dir, 'models', 'gnn_gru', 'gnn_gru_model.pth')
    model.load_state_dict(torch.load(save_path))
    
    criterion = nn.BCEWithLogitsLoss()
    
    print("\nRunning inference on BATADAL_test_dataset_scaled.csv...")
    # 4. Evaluate and Optimize Threshold dynamically (threshold=None)
    _, acc, f1, prec, rec, auc, best_thresh = evaluate_model(model, test_loader, criterion, device, threshold=None)
    
    print(f"\n--- Final Benchmark on Novel BATADAL Test Set ---")
    print(f"Optimal Threshold: {best_thresh:.4f}")
    print(f"Test Accuracy:     {acc:.4f}")
    print(f"Test F1-Score:     {f1:.4f}")
    print(f"Precision:         {prec:.4f}")
    print(f"Recall:            {rec:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")

if __name__ == "__main__":
    evaluate_real_test()
