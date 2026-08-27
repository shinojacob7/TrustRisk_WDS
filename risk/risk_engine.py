import networkx as nx
import numpy as np
import pickle
import pandas as pd
import os

class RiskEngine:
    """
    Phase 2: Cyber-Physical Network & Physics Layer (Risk Engine)
    Computes mathematically rigorous risk indices for Water Distribution Systems (WDS).
    Replaces heuristic proxies with formal topological and hydraulic formulations.
    """
    def __init__(self, graph_path):
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
            
        with open(graph_path, 'rb') as f:
            self.G = pickle.load(f)
            
        # Topological metrics
        self.node_betweenness = nx.betweenness_centrality(self.G)
        self.node_closeness = nx.closeness_centrality(self.G)
        self.node_degree = nx.degree_centrality(self.G)
        self.edge_betweenness = nx.edge_betweenness_centrality(self.G)
        
        # Precompute shortest paths for cascading risk (CRI)
        # Using unweighted shortest paths as a proxy for hydraulic reachability
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(self.G))

    def compute_asset_criticality(self):
        """
        MDCI (Multi-Dimensional Criticality Index).
        Formal formulation: A weighted aggregation of structural centralities representing
        the node's topological importance in maintaining network connectivity.
        MDCI_i = w1 * Betweenness_i + w2 * Closeness_i + w3 * Degree_i
        """
        mdci_nodes = {}
        for node in self.G.nodes():
            structural_crit = (self.node_betweenness[node] * 0.4 + 
                               self.node_closeness[node] * 0.4 + 
                               self.node_degree[node] * 0.2)
            mdci_nodes[node] = structural_crit
            
        mdci_edges = {}
        for u, v in self.G.edges():
            b_cent = self.edge_betweenness.get((u, v), self.edge_betweenness.get((v, u), 0))
            mdci_edges[(u, v)] = b_cent
            
        # Normalize to [0, 1]
        max_n = max(mdci_nodes.values()) + 1e-9
        mdci_nodes = {k: v/max_n for k,v in mdci_nodes.items()}
        
        return mdci_nodes, mdci_edges

    def compute_attack_susceptibility(self, observed_nodes):
        """
        ASS (Attack Susceptibility Score).
        Formal formulation: Quantifies cyber-exposure. 
        - Actuators (Pumps/Valves) have highest susceptibility (can be overridden remotely).
        - Monitored sensors (PLCs) have medium susceptibility (can be spoofed).
        - Unmonitored passive components have lowest susceptibility.
        """
        ass_nodes = {}
        for node, data in self.G.nodes(data=True):
            node_type = data.get('node_type', 'Junction')
            if node_type in ['Tank', 'Reservoir']:
                ass_nodes[node] = 0.8  # Actuator/Storage control
            elif node in observed_nodes:
                ass_nodes[node] = 0.6  # PLC/Sensor exposed
            else:
                ass_nodes[node] = 0.1  # Passive unmonitored junction
                
        return ass_nodes

    def compute_impact_severity(self):
        """
        ISS (Impact Severity Score).
        Formal formulation: The immediate hydraulic consequence of asset failure.
        ISS_i = Demand_i / Max_Demand (for consumption nodes)
        Tanks/Reservoirs assigned ISS = 1.0 due to systemic source disruption.
        """
        iss_nodes = {}
        demands = []
        for n, d in self.G.nodes(data=True):
            val = d.get('base_demand')
            if val is not None:
                demands.append(float(val))
                
        max_demand = max(demands + [1e-9])
        
        for node, data in self.G.nodes(data=True):
            val = data.get('base_demand')
            demand = float(val) if val is not None else 0.0
            node_type = data.get('node_type', 'Junction')
            
            if node_type in ['Tank', 'Reservoir']:
                iss_nodes[node] = 1.0  
            else:
                iss_nodes[node] = (demand / max_demand) if demand > 0 else 0.05
                
        return iss_nodes

    def compute_cascading_risk(self, iss_nodes):
        """
        CRI (Cascading Risk Index).
        Formal formulation: Hydraulic Reachability Impact.
        Models how disruption propagates. A node's failure affects downstream nodes inversely 
        proportional to topological distance.
        CRI_i = Sum_{j} (ISS_j / d(i,j)^2)
        """
        cri_nodes = {}
        for i in self.G.nodes():
            cascade_score = 0.0
            if i in self.shortest_paths:
                for j, dist in self.shortest_paths[i].items():
                    if dist > 0:
                        # Influence decays quadratically with topological distance
                        cascade_score += iss_nodes.get(j, 0) / (dist ** 2)
            cri_nodes[i] = cascade_score
            
        # Normalize
        max_cri = max(cri_nodes.values()) + 1e-9
        cri_nodes = {k: v/max_cri for k,v in cri_nodes.items()}
        
        return cri_nodes

    def generate_risk_report(self, observed_nodes):
        mdci_n, _ = self.compute_asset_criticality()
        ass_n = self.compute_attack_susceptibility(observed_nodes)
        iss_n = self.compute_impact_severity()
        cri_n = self.compute_cascading_risk(iss_n)
        
        data = []
        for node, attrs in self.G.nodes(data=True):
            data.append({
                'Node_ID': node,
                'Type': attrs.get('node_type', 'Junction'),
                'Criticality_MDCI': mdci_n.get(node, 0),
                'Susceptibility_ASS': ass_n.get(node, 0),
                'Impact_ISS': iss_n.get(node, 0),
                'Cascading_CRI': cri_n.get(node, 0)
            })
            
        df = pd.DataFrame(data)
        # Unified multiplication representing the chain of vulnerability (Prob * Consequence)
        df['Baseline_Risk_Score'] = df['Criticality_MDCI'] * df['Susceptibility_ASS'] * df['Impact_ISS'] * df['Cascading_CRI']
        
        # Max-Min scaling for the final physical baseline
        max_risk = df['Baseline_Risk_Score'].max() + 1e-9
        df['Baseline_Risk_Score'] = df['Baseline_Risk_Score'] / max_risk
        
        return df

if __name__ == "__main__":
    # Test formal Risk Engine
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(base_dir, 'network', 'graph', 'ctown_graph.pkl')
    
    engine = RiskEngine(graph_path)
    
    # Mock observed sensors based on BATADAL (T1-T7, J280, etc.)
    observed = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'J280', 'J269', 'J300', 'J256', 'J289', 'J415', 'J302', 'J306', 'J307', 'J317', 'J14', 'J422']
    
    risk_df = engine.generate_risk_report(observed)
    
    print("\n--- Top 5 Most Vulnerable Assets (Formal Mathematical Baseline) ---")
    print(risk_df.sort_values(by='Baseline_Risk_Score', ascending=False).head(5))
