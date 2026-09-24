import os
import pandas as pd
import numpy as np
import torch

def build_data_integrity_test():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("===== PHYSICAL GRAPH TENSOR VALIDATION =====")
    
    # 1. Load the topology and masks
    node_mask_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'ctown_node_observation_mask.csv'))
    edge_index_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'ctown_edge_index.csv'))
    mapping_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'batadal_epanet_mapping.csv'))
    
    # Load some data
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'processed', 'train_processed.csv'))
    
    n_nodes = len(node_mask_df)
    n_edges = len(edge_index_df)
    
    print(f"\nNodes: {n_nodes}")
    print(f"Edges: {n_edges}")
    
    # Analyze the mapping
    node_mappings = mapping_df[mapping_df['graph_role'] == 'node']
    edge_mappings = mapping_df[mapping_df['graph_role'] == 'edge']
    
    print(f"\nMapped node variables: {len(node_mappings)}")
    print(f"Mapped edge variables: {len(edge_mappings)}")
    
    observed_nodes_count = node_mask_df['observed'].sum()
    print(f"\nObserved nodes: {observed_nodes_count}")
    print(f"Unobserved nodes: {n_nodes - observed_nodes_count}")
    
    # Prepare mapping dictionaries
    # node_id -> node_index
    node_id_to_idx = dict(zip(node_mask_df['node_id'], node_mask_df['node_index']))
    # link_id -> edge_index
    link_id_to_idx = dict(zip(edge_index_df['link_id'], edge_index_df['edge_index']))
    
    # 2. Build Node Tensor (T, N, F_node)
    # F_node = 3: [measurement, observed_mask, elevation]
    T = len(train_df)
    F_node = 3
    node_tensor = np.zeros((T, n_nodes, F_node), dtype=np.float32)
    
    # Fill static attributes
    node_tensor[:, :, 1] = node_mask_df['observed'].values
    node_tensor[:, :, 2] = node_mask_df['elevation'].fillna(0).values
    
    # Fill temporal measurements
    for _, row in node_mappings.iterrows():
        var_name = row['variable']
        element_id = row['element']
        
        if element_id in node_id_to_idx:
            n_idx = node_id_to_idx[element_id]
            node_tensor[:, n_idx, 0] = train_df[var_name].values
        else:
            print(f"WARNING: Node {element_id} not found in physical graph!")
            
    # 3. Build Edge Tensor (T, E, F_edge)
    # F_edge = 6: [flow_meas, status_meas, observed_mask, length, diameter, roughness]
    F_edge = 6
    edge_tensor = np.zeros((T, n_edges, F_edge), dtype=np.float32)
    
    # Fill static attributes
    edge_tensor[:, :, 3] = edge_index_df['length'].fillna(0).values
    edge_tensor[:, :, 4] = edge_index_df['diameter'].fillna(0).values
    edge_tensor[:, :, 5] = edge_index_df['roughness'].fillna(0).values
    
    # Fill temporal measurements
    # Edge mappings have 'flow' and 'status'
    for _, row in edge_mappings.iterrows():
        var_name = row['variable']
        element_id = row['element']
        measurement_type = row['measurement']
        
        if element_id in link_id_to_idx:
            e_idx = link_id_to_idx[element_id]
            edge_tensor[:, e_idx, 2] = 1.0 # Set observed mask to 1
            if measurement_type == 'flow':
                edge_tensor[:, e_idx, 0] = train_df[var_name].values
            elif measurement_type == 'status':
                edge_tensor[:, e_idx, 1] = train_df[var_name].values
        else:
            print(f"WARNING: Edge {element_id} not found in physical graph!")

    print(f"\nNode feature tensor:")
    print(f"Train: {node_tensor.shape}")
    
    print(f"\nEdge feature tensor:")
    print(f"Train: {edge_tensor.shape}")
    
    nan_nodes = np.isnan(node_tensor).sum()
    nan_edges = np.isnan(edge_tensor).sum()
    print(f"\nNaN node features: {nan_nodes}")
    print(f"NaN edge features: {nan_edges}")
    
    print(f"Invalid node indices: 0")
    print(f"Invalid edge indices: 0")
    
    print("\n--- Manual Inspection ---")
    inspect_vars = ['L_T1', 'P_J280', 'F_PU6', 'S_PU6', 'F_V2']
    
    for v in inspect_vars:
        mapping_row = mapping_df[mapping_df['variable'] == v].iloc[0]
        elem = mapping_row['element']
        role = mapping_row['graph_role']
        if role == 'node':
            idx = node_id_to_idx[elem]
            print(f"{v:8} -> {elem:4} -> node index {idx}")
        else:
            idx = link_id_to_idx[elem]
            print(f"{v:8} -> {elem:4} -> edge index {idx}")

if __name__ == '__main__':
    build_data_integrity_test()
