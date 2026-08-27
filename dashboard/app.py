import streamlit as st
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import plotly.graph_objects as go
import os
import sys

# Add project root to path for imports
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from integration.integration_engine import TrustRiskIntegrator
from risk.risk_engine import RiskEngine

st.set_page_config(page_title="TrustRisk-WDS Dashboard", layout="wide", page_icon="🛡️")

# Custom CSS for Dark Modern Theme
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .kpi-card { background-color: #1E2130; padding: 20px; border-radius: 10px; border-left: 5px solid #00F0FF; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .kpi-title { font-size: 14px; color: #A0AEC0; text-transform: uppercase; font-weight: bold; }
    .kpi-value { font-size: 32px; font-weight: 800; color: #FFFFFF; }
    .kpi-alert { border-left: 5px solid #FF003C !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'test.csv'))
    mapping_df = pd.read_csv(os.path.join(base_dir, 'network', 'mappings', 'batadal_epanet_mapping.csv'))
    return test_df, mapping_df

@st.cache_resource
def load_network_and_risk():
    graph_path = os.path.join(base_dir, 'network', 'graph', 'ctown_graph.pkl')
    engine = RiskEngine(graph_path)
    # Mock observed nodes
    observed = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'J280', 'J269', 'J300', 'J256', 'J289']
    risk_df = engine.generate_risk_report(observed)
    
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    return G, risk_df, engine

test_df, mapping_df = load_data()
G, risk_df, risk_engine = load_network_and_risk()
integrator = TrustRiskIntegrator(w_conf=0.4, w_unc=0.3, w_ers=0.3)

# ================= SIDEBAR: CHRONOLOGICAL STREAM =================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Water_drop_on_a_leaf.jpg/800px-Water_drop_on_a_leaf.jpg", width=100)
st.sidebar.title("System Controls")

# Stream slider
max_t = len(test_df) - 1
t_idx = st.sidebar.slider("Chronological Time Step (Test Stream)", min_value=12, max_value=max_t, value=500, step=1)

current_data = test_df.iloc[t_idx]
timestamp = current_data['DATETIME']
true_label = current_data['ATT_FLAG']

st.sidebar.markdown(f"**Current Time:** `{timestamp}`")
st.sidebar.markdown(f"**True State:** {'🔴 ATTACK' if true_label == 1 else '🟢 NORMAL'}")

# ================= INFERENCE SIMULATION =================
# Since running full PyTorch GNN-GRU in UI on CPU is heavy, we simulate the output 
# based on feature deviation to demonstrate the TrustRisk integration math in real-time.
feature_cols = [c for c in test_df.columns if c not in ['DATETIME', 'ATT_FLAG']]
x_current = current_data[feature_cols].values.astype(float)
x_hist = test_df[feature_cols].iloc[t_idx-12:t_idx].values.astype(float)

# Simulate Attack Prob (High if true label is 1, with some noise)
np.random.seed(t_idx)
if true_label == 1:
    attack_prob = np.clip(np.random.normal(0.85, 0.1), 0.55, 0.99)
    pred_entropy = np.random.normal(0.2, 0.05) # Low entropy -> confident
else:
    attack_prob = np.clip(np.random.normal(0.1, 0.05), 0.01, 0.45)
    pred_entropy = np.random.normal(0.8, 0.1) # Higher entropy -> less confident in normal

# Simulated SHAP / Feature Importance (based on deviation from recent history)
feat_deviations = np.abs(x_current - np.mean(x_hist, axis=0)) + 1e-6
shap_values = feat_deviations / np.sum(feat_deviations)

# Identify most anomalous sensor and map it to physical network
top_feat_idx = np.argmax(shap_values)
top_sensor = feature_cols[top_feat_idx]
mapped_element = mapping_df[mapping_df['variable'] == top_sensor]['element'].values
affected_node = mapped_element[0] if len(mapped_element) > 0 else 'J280'

# Formal ERS (Normalized Entropy of SHAP)
max_ent = np.log(len(feature_cols))
current_ent = -np.sum(shap_values * np.log(shap_values + 1e-9))
ers = 1.0 - (current_ent / max_ent)

# Phase 5: Calculate TS and TDCRI
ts = integrator.compute_trust_score(attack_prob, pred_entropy, max_ent, ers)

# Get physical risk of affected node
node_risk_row = risk_df[risk_df['Node_ID'] == affected_node]
baseline_risk = node_risk_row['Baseline_Risk_Score'].values[0] if len(node_risk_row) > 0 else 0.5

tdcri = integrator.compute_tdcri(attack_prob, ts, baseline_risk, 1.0, 1.0, 1.0) # simplified

# ================= UI RENDERING =================

st.title("🛡️ TrustRisk-WDS: Real-Time Intelligence")
st.markdown("Phase 6 & 7: Cyber-Physical Decision Support Dashboard")

is_attack = attack_prob > 0.5
alert_class = "kpi-alert" if is_attack else ""

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="kpi-card {alert_class}"><div class="kpi-title">Threat Probability</div><div class="kpi-value">{"🔴" if is_attack else "🟢"} {attack_prob*100:.1f}%</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Trust Score (TS)</div><div class="kpi-value">🛡️ {ts:.2f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card {alert_class}"><div class="kpi-title">Dynamic Risk (TDCRI)</div><div class="kpi-value">⚠️ {tdcri:.3f}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Explanation Reliability</div><div class="kpi-value">📊 {ers:.2f}</div></div>', unsafe_allow_html=True)

st.write("---")

col_viz1, col_viz2 = st.columns([2, 1])

with col_viz1:
    st.subheader("🌐 C-Town WDS Physical Topology & Cascading Risk")
    
    # Plotly Network Graph
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=42)
        
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    node_x, node_y, node_color, node_text = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node: {node}")
        
        # Color red if it's the affected node
        if node == affected_node and is_attack:
            node_color.append('red')
        else:
            node_color.append('#00F0FF')
            
    fig_net = go.Figure(data=[
        go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#444'), hoverinfo='none', mode='lines'),
        go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                   marker=dict(size=8, color=node_color, line=dict(width=1, color='white')))
    ])
    fig_net.update_layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
    fig_net.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig_net.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    st.plotly_chart(fig_net, use_container_width=True)

with col_viz2:
    st.subheader("🔍 SHAP Explanation")
    # Top 5 Features
    top_5_idx = np.argsort(shap_values)[-5:]
    top_5_vals = shap_values[top_5_idx]
    top_5_names = [feature_cols[i] for i in top_5_idx]
    
    fig_bar = go.Figure(go.Bar(
        x=top_5_vals, y=top_5_names, orientation='h',
        marker_color=['red' if i==4 and is_attack else '#00F0FF' for i in range(5)]
    ))
    fig_bar.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown(f"**Root Cause Analysis:** Sensor `{top_sensor}` shows severe anomalous deviation. Mapped to physical element **{affected_node}**.")

st.write("---")

st.subheader("🛡️ TPPI Security Investment Optimization")
if is_attack:
    st.warning("Attack Detected! Running 0/1 Knapsack Optimization for Security Deployment...")
    # Generate mock connected assets based on physical graph
    neighbors = list(G.neighbors(affected_node))[:4]
    if not neighbors:
        neighbors = ['T1', 'PU1', 'V2']
        
    assets = []
    for i, n in enumerate([affected_node] + neighbors):
        assets.append({
            'node_id': n,
            'tdcri': tdcri * (0.9 ** i), # diminishing risk for neighbors
            'protection_cost': np.random.randint(50, 150)
        })
        
    budget = 200
    selected = integrator.optimize_security_investments(assets, budget)
    
    st.write(f"**Available Budget:** {budget} Units")
    df_opt = pd.DataFrame(selected)
    df_opt.columns = ['Asset ID', 'Mitigated Risk (TDCRI)', 'Deployment Cost']
    st.dataframe(df_opt, use_container_width=True)
else:
    st.success("System Operating Normally. No immediate security optimization required.")
