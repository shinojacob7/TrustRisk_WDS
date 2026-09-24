import os
import pandas as pd
import networkx as nx

def build_actuator_anchored_graph():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mappings_dir = os.path.join(base_dir, 'network', 'mappings')
    
    node_df = pd.read_csv(os.path.join(mappings_dir, 'ctown_node_observation_mask.csv'))
    edge_df = pd.read_csv(os.path.join(mappings_dir, 'ctown_edge_index.csv'))
    mapping_df = pd.read_csv(os.path.join(mappings_dir, 'batadal_epanet_mapping.csv'))
    
    sensor_nodes = set(node_df[node_df['observed'] == 1]['node_id'].tolist())
    
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['start_node'], row['end_node'], **row.to_dict())
        
    actuator_links = edge_df[edge_df['link_type'].isin(['Pump', 'Valve'])]['link_id'].tolist()
    
    condensed_edges = []
    mapped_actuators = set()
    
    # Helper to find nearest sensor in a given direction (avoiding the actuator itself)
    def find_nearest_sensor(start_node, forbidden_link_id):
        if start_node in sensor_nodes:
            return start_node, 0.0, 0
            
        queue = [(start_node, 0.0, 0)]
        visited = {start_node}
        
        while queue:
            # Sort queue by distance to ensure deterministic shortest path
            queue.sort(key=lambda x: x[1])
            curr, dist, pipes = queue.pop(0)
            
            # Sort neighbors for determinism
            neighbors = sorted(list(G.neighbors(curr)))
            for neighbor in neighbors:
                edata = G.get_edge_data(curr, neighbor)
                if edata['link_id'] == forbidden_link_id:
                    continue
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    l = edata['length'] if not pd.isna(edata['length']) else 1.0
                    new_dist = dist + l
                    new_pipes = pipes + (1 if edata['link_type'] == 'Pipe' else 0)
                    
                    if neighbor in sensor_nodes:
                        return neighbor, new_dist, new_pipes
                    else:
                        queue.append((neighbor, new_dist, new_pipes))
        return None, 0.0, 0

    # 1. Map all 15 Actuators
    for act_id in actuator_links:
        act_data = edge_df[edge_df['link_id'] == act_id].iloc[0]
        u, v = act_data['start_node'], act_data['end_node']
        
        s_u, dist_u, pipes_u = find_nearest_sensor(u, act_id)
        s_v, dist_v, pipes_v = find_nearest_sensor(v, act_id)
        
        if s_u and s_v:
            condensed_edges.append({
                'source': s_u,
                'target': s_v,
                'edge_type': 'Actuator',
                'link_id': act_id,
                'link_type': act_data['link_type'],
                'physical_path_length': dist_u + dist_v + (act_data['length'] if not pd.isna(act_data['length']) else 0.0),
                'num_intermediate_nodes': pipes_u + pipes_v, # approximate
                'num_pipes': pipes_u + pipes_v
            })
            mapped_actuators.add(act_id)

    # 2. Map pure Pipe connections (paths between sensors containing NO actuators)
    for s_start in sorted(list(sensor_nodes)):
        queue = [(s_start, 0.0, 0)]
        visited = {s_start}
        
        while queue:
            queue.sort(key=lambda x: x[1])
            curr, dist, pipes = queue.pop(0)
            
            for neighbor in sorted(list(G.neighbors(curr))):
                edata = G.get_edge_data(curr, neighbor)
                
                # Do not cross actuators for pure pipe paths
                if edata['link_type'] in ['Pump', 'Valve']:
                    continue
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_dist = dist + edata['length']
                    new_pipes = pipes + 1
                    
                    if neighbor in sensor_nodes:
                        # Found a pure pipe path to another sensor
                        # Ensure we don't add duplicates (undirected)
                        if s_start < neighbor:
                            condensed_edges.append({
                                'source': s_start,
                                'target': neighbor,
                                'edge_type': 'Pipe',
                                'link_id': f"PATH_{s_start}_{neighbor}",
                                'link_type': 'Pipe',
                                'physical_path_length': new_dist,
                                'num_intermediate_nodes': new_pipes - 1,
                                'num_pipes': new_pipes
                            })
                    else:
                        queue.append((neighbor, new_dist, new_pipes))

    edges_df = pd.DataFrame(condensed_edges)
    nodes_df = pd.DataFrame({'node_id': sorted(list(sensor_nodes))})
    
    edges_df.to_csv(os.path.join(mappings_dir, 'sensor_graph_edges.csv'), index=False)
    nodes_df.to_csv(os.path.join(mappings_dir, 'sensor_graph_nodes.csv'), index=False)
    
    # ================= VALIDATION REPORT =================
    print("===== SENSOR GRAPH VALIDATION =====")
    print(f"Sensor nodes: {len(nodes_df)} / 19")
    print(f"Condensed edges: {len(edges_df)}")
    
    pumps = edge_df[edge_df['link_type'] == 'Pump']['link_id'].tolist()
    valves = edge_df[edge_df['link_type'] == 'Valve']['link_id'].tolist()
    
    print("\nPumps:")
    pump_count = 0
    for p in pumps:
        status = "[OK]" if p in mapped_actuators else "[MISSING]"
        if status == "[OK]": pump_count += 1
        print(f"{p:5} {status}")
    print(f"\nPump coverage: {pump_count} / {len(pumps)}")
    
    print("\nValves:")
    valve_count = 0
    for v in valves:
        status = "[OK]" if v in mapped_actuators else "[MISSING]"
        if status == "[OK]": valve_count += 1
        print(f"{v:5} {status}")
    print(f"\nValve coverage: {valve_count} / {len(valves)}")
    
    print(f"\nTotal actuator coverage: {len(mapped_actuators)} / {len(actuator_links)}")
    
    # Validate 43 variables
    edge_mappings = mapping_df[mapping_df['graph_role'] == 'edge']
    node_mappings = mapping_df[mapping_df['graph_role'] == 'node']
    
    represented_vars = set()
    for _, row in node_mappings.iterrows():
        if row['element'] in sensor_nodes:
            represented_vars.add(row['variable'])
            
    for _, row in edge_mappings.iterrows():
        if row['element'] in mapped_actuators:
            represented_vars.add(row['variable'])
            
    total_vars = len(mapping_df)
    
    print(f"\nOriginal SCADA variables: {total_vars}")
    print(f"Represented in Model E: {len(represented_vars)}")
    print(f"Missing variables: {total_vars - len(represented_vars)}")
    
    if total_vars - len(represented_vars) > 0:
        missing = set(mapping_df['variable']) - represented_vars
        print(f"Missing: {missing}")

if __name__ == '__main__':
    build_actuator_anchored_graph()
