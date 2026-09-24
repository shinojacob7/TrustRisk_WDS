import os

with open('dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We will patch the dashboard to correctly assign high risk to actuators (Pumps/Valves)
old_risk_logic = """node_risk_row = risk_df[risk_df['Node_ID'] == affected_node]
baseline_risk = node_risk_row['Baseline_Risk_Score'].values[0] if len(node_risk_row) > 0 else 0.5"""

new_risk_logic = """node_risk_row = risk_df[risk_df['Node_ID'] == affected_node]
if len(node_risk_row) > 0:
    baseline_risk = node_risk_row['Baseline_Risk_Score'].values[0]
else:
    # If it's not in the node list, it's an Edge (Pump or Valve).
    # Actuators are the most critical components for cyber-physical security!
    baseline_risk = 0.95 if 'PU' in affected_node or 'V' in affected_node else 0.5"""

content = content.replace(old_risk_logic, new_risk_logic)

with open('dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Dashboard Risk Mapping patched successfully!")
