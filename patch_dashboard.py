import os

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update load_data
content = content.replace(
    "    return test_df, mapping_df",
    "    trust_df = pd.read_csv(os.path.join(base_dir, 'experiments', 'final', 'trust_pipeline.csv'))\n    return test_df, mapping_df, trust_df"
)
content = content.replace("test_df, mapping_df = load_data()", "test_df, mapping_df, trust_df = load_data()")

# 2. Update slider min value and add trust_row extraction
old_slider = """# Stream slider
max_t = len(test_df) - 1
t_idx = st.sidebar.slider("Chronological Time Step (Test Stream)", min_value=12, max_value=max_t, value=500, step=1)"""
new_slider = """# Stream slider
min_t = 11
max_t = len(test_df) - 1
t_idx = st.sidebar.slider("Chronological Time Step (Test Stream)", min_value=min_t, max_value=max_t, value=500, step=1)

# Map physical time to trust pipeline index
trust_idx = t_idx - min_t
trust_row = trust_df.iloc[trust_idx]"""
content = content.replace(old_slider, new_slider)

# 3. Replace the entire INFERENCE SIMULATION block
old_sim = """# ================= INFERENCE SIMULATION =================
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
ts = integrator.compute_trust_score(attack_prob, pred_entropy, max_ent, ers)"""

new_sim = """# ================= REAL TRUST PIPELINE =================
# We load the exact pre-calculated metrics from the frozen Trust Pipeline
attack_prob = trust_row['p_attack']
confidence = trust_row['C']
u_norm = trust_row['U_norm']
ecp = trust_row['ECP']
ts = trust_row['Trust_Score']

# For visualization purposes only, we quickly estimate the localized feature deviation
# since the full 396x12 SHAP matrices were too large to cache in the CSV.
feature_cols = [c for c in test_df.columns if c not in ['DATETIME', 'ATT_FLAG']]
x_current = current_data[feature_cols].values.astype(float)
x_hist = test_df[feature_cols].iloc[t_idx-11:t_idx].values.astype(float) # 11 for historical context
feat_deviations = np.abs(x_current - np.mean(x_hist, axis=0)) + 1e-6
shap_values = feat_deviations / np.sum(feat_deviations)

top_feat_idx = np.argmax(shap_values)
top_sensor = feature_cols[top_feat_idx]
mapped_element = mapping_df[mapping_df['variable'] == top_sensor]['element'].values
affected_node = mapped_element[0] if len(mapped_element) > 0 else 'J280'"""
content = content.replace(old_sim, new_sim)

# 4. Update TDCRI calculation to the new math
old_tdcri = "tdcri = integrator.compute_tdcri(attack_prob, ts, baseline_risk, 1.0, 1.0, 1.0) # simplified"
new_tdcri = "tdcri = (attack_prob * ts) * baseline_risk # TDCRI = Trusted Prob * Risk Magnitude"
content = content.replace(old_tdcri, new_tdcri)

# 5. Update UI labels
content = content.replace("Explanation Reliability", "Explanation Concentration (ECP)")
content = content.replace("{ers:.2f}", "{ecp:.2f}")

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Dashboard updated successfully!")
