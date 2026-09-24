import os
import pandas as pd

def validate_model_e_variables():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mappings_dir = os.path.join(base_dir, 'network', 'mappings')
    
    nodes_df = pd.read_csv(os.path.join(mappings_dir, 'sensor_graph_nodes.csv'))
    edges_df = pd.read_csv(os.path.join(mappings_dir, 'sensor_graph_edges.csv'))
    mapping_df = pd.read_csv(os.path.join(mappings_dir, 'batadal_epanet_mapping.csv'))
    
    # 1. Tally original variables
    orig_vars = set(mapping_df['variable'].tolist())
    
    # 2. Tally Model E variables
    model_e_vars = set()
    
    # Node variables
    sensor_nodes = set(nodes_df['node_id'].tolist())
    node_mappings = mapping_df[mapping_df['graph_role'] == 'node']
    for _, row in node_mappings.iterrows():
        if row['element'] in sensor_nodes:
            model_e_vars.add(row['variable'])
            
    # Edge variables (Actuators)
    actuator_edges = edges_df[edges_df['edge_type'] == 'Actuator']
    mapped_actuators = set(actuator_edges['link_id'].tolist())
    
    edge_mappings = mapping_df[mapping_df['graph_role'] == 'edge']
    pump_vars_found = 0
    valve_vars_found = 0
    
    for _, row in edge_mappings.iterrows():
        if row['element'] in mapped_actuators:
            model_e_vars.add(row['variable'])
            if 'PU' in row['element']:
                pump_vars_found += 1
            else:
                valve_vars_found += 1
                
    missing = orig_vars - model_e_vars
    extra = model_e_vars - orig_vars
    
    print("Original BATADAL variables:       43")
    print(f"Model E variables:                {len(model_e_vars)}")
    print(f"Missing BATADAL variables:         {len(missing)}")
    print(f"Extra variables:                   {len(extra)}")
    print("")
    print(f"Observed nodes:                   {len(sensor_nodes)}/19")
    print(f"Pump variables:                   {pump_vars_found}/22")
    print(f"Valve variables:                   {valve_vars_found}/2")
    print("")
    print(f"Sensor graph nodes:               {len(nodes_df)}")
    print(f"Sensor graph edges:               {len(edges_df)}")

if __name__ == '__main__':
    validate_model_e_variables()
