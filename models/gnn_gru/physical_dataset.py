import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class WDSPhysicalGraphDataset(Dataset):
    """
    Constructs temporal windows of the physical C-Town WDS graph.
    Node features: [measurement, observed_mask, elevation, has_elevation]
    Edge features: [flow, status, observed_mask, length, diameter, roughness, has_length, has_diameter, has_roughness, is_pipe, is_pump, is_valve]
    """
    def __init__(self, df, base_dir, window_size=12, target_col='ATT_FLAG', bidirectional=True, use_physical_attributes=True):
        self.df = df.copy()
        self.window_size = window_size
        self.target_col = target_col
        self.base_dir = base_dir
        self.bidirectional = bidirectional
        self.use_physical_attributes = use_physical_attributes
        
        self.labels = self.df[target_col].values.astype(np.float32)
        
        self._load_topology()
        self._build_tensors()
        self._build_edge_index()

    def _load_topology(self):
        self.node_mask_df = pd.read_csv(os.path.join(self.base_dir, 'network', 'mappings', 'ctown_node_observation_mask.csv'))
        self.edge_index_df = pd.read_csv(os.path.join(self.base_dir, 'network', 'mappings', 'ctown_edge_index.csv'))
        self.mapping_df = pd.read_csv(os.path.join(self.base_dir, 'network', 'mappings', 'batadal_epanet_mapping.csv'))
        
        self.n_nodes = len(self.node_mask_df)
        self.n_edges = len(self.edge_index_df)
        
        self.node_id_to_idx = dict(zip(self.node_mask_df['node_id'], self.node_mask_df['node_index']))
        self.link_id_to_idx = dict(zip(self.edge_index_df['link_id'], self.edge_index_df['edge_index']))
        
        self.node_mappings = self.mapping_df[self.mapping_df['graph_role'] == 'node']
        self.edge_mappings = self.mapping_df[self.mapping_df['graph_role'] == 'edge']

    def _build_tensors(self):
        T = len(self.df)
        
        # --- Node Tensor ---
        self.F_node = 4 # [meas, mask, elev_scaled, has_elev]
        self.node_tensor = np.zeros((T, self.n_nodes, self.F_node), dtype=np.float32)
        
        self.node_tensor[:, :, 1] = self.node_mask_df['observed'].values
        
        # Step 1 & 2: Normalize elevation and add applicability mask
        elevations = self.node_mask_df['elevation'].values
        has_elev = ~np.isnan(elevations)
        elev_safe = np.where(has_elev, elevations, 0.0)
        elev_mean = elev_safe[has_elev].mean() if has_elev.sum() > 0 else 0.0
        elev_std = elev_safe[has_elev].std() if has_elev.sum() > 0 else 1.0
        if elev_std == 0: elev_std = 1.0
        elev_scaled = np.where(has_elev, (elev_safe - elev_mean) / elev_std, 0.0)
        
        if self.use_physical_attributes:
            self.node_tensor[:, :, 2] = elev_scaled
            self.node_tensor[:, :, 3] = has_elev.astype(np.float32)
        
        for _, row in self.node_mappings.iterrows():
            if row['element'] in self.node_id_to_idx:
                n_idx = self.node_id_to_idx[row['element']]
                self.node_tensor[:, n_idx, 0] = self.df[row['variable']].values
                
        # --- Edge Tensor ---
        self.F_edge = 12 # [flow, status, mask, len_s, diam_s, rough_s, has_len, has_diam, has_rough, is_pipe, is_pump, is_valve]
        self.edge_tensor = np.zeros((T, self.n_edges, self.F_edge), dtype=np.float32)
        
        link_types = self.edge_index_df['link_type'].values
        is_pipe = (link_types == 'Pipe')
        is_pump = (link_types == 'Pump')
        is_valve = (link_types == 'Valve')
        
        self.edge_tensor[:, :, 9] = is_pipe.astype(np.float32)
        self.edge_tensor[:, :, 10] = is_pump.astype(np.float32)
        self.edge_tensor[:, :, 11] = is_valve.astype(np.float32)
        
        # Step 1 & 2: Normalize physical edge attributes and add applicability masks
        def normalize_attr(attr_name):
            raw = self.edge_index_df[attr_name].values
            has_attr = ~np.isnan(raw)
            safe = np.where(has_attr, raw, 0.0)
            mean_val = safe[has_attr].mean() if has_attr.sum() > 0 else 0.0
            std_val = safe[has_attr].std() if has_attr.sum() > 0 else 1.0
            if std_val == 0: std_val = 1.0
            scaled = np.where(has_attr, (safe - mean_val) / std_val, 0.0)
            return scaled, has_attr.astype(np.float32)
            
        len_s, has_len = normalize_attr('length')
        diam_s, has_diam = normalize_attr('diameter')
        rough_s, has_rough = normalize_attr('roughness')
        
        if self.use_physical_attributes:
            self.edge_tensor[:, :, 3] = len_s
            self.edge_tensor[:, :, 4] = diam_s
            self.edge_tensor[:, :, 5] = rough_s
            self.edge_tensor[:, :, 6] = has_len
            self.edge_tensor[:, :, 7] = has_diam
            self.edge_tensor[:, :, 8] = has_rough
        
        for _, row in self.edge_mappings.iterrows():
            if row['element'] in self.link_id_to_idx:
                e_idx = self.link_id_to_idx[row['element']]
                self.edge_tensor[:, e_idx, 2] = 1.0 # observed
                if row['measurement'] == 'flow':
                    self.edge_tensor[:, e_idx, 0] = self.df[row['variable']].values
                elif row['measurement'] == 'status':
                    self.edge_tensor[:, e_idx, 1] = self.df[row['variable']].values

    def _build_edge_index(self):
        src = self.edge_index_df['start_index'].values
        dst = self.edge_index_df['end_index'].values
        
        edge_index = np.vstack((src, dst))
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        if self.bidirectional:
            rev_edge_index = np.vstack((dst, src))
            self.edge_index = torch.tensor(np.hstack((edge_index, rev_edge_index)), dtype=torch.long)
            self.edge_tensor = np.concatenate([self.edge_tensor, self.edge_tensor], axis=1)

    def __len__(self):
        return max(0, len(self.df) - self.window_size + 1)

    def __getitem__(self, idx):
        X_window = torch.tensor(self.node_tensor[idx : idx + self.window_size])
        E_window = torch.tensor(self.edge_tensor[idx : idx + self.window_size])
        y = torch.tensor(self.labels[idx + self.window_size - 1])
        return X_window, E_window, self.edge_index, y

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'train_processed.csv'))
    
    ds = WDSPhysicalGraphDataset(train_df, base_dir, window_size=12, bidirectional=True)
    X_w, E_w, edge_idx, y = ds[0]
    
    print("=== PyTorch Geometric Graph Window Validation ===")
    print(f"X_window shape: {X_w.shape} (Window, Nodes, Features={ds.F_node})")
    print(f"E_window shape: {E_w.shape} (Window, Edges, Features={ds.F_edge})")
    print(f"edge_index shape: {edge_idx.shape} (2, Edges)")
    print(f"Label: {y.item()}")
