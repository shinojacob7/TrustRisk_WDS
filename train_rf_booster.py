import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os
import pickle
import numpy as np

# Load data
train_df = pd.read_csv('data/splits/train_multiple.csv')
test_df = pd.read_csv('data/splits/test.csv')

features = [c for c in train_df.columns if c not in ['DATETIME', 'ATT_FLAG']]

X_train = train_df[features]
y_train = train_df['ATT_FLAG']

X_test = test_df[features]
y_test = test_df['ATT_FLAG']

# Train Random Forest Booster
print("Training RF Booster...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict
p_rf = rf.predict_proba(X_test)[:, 1]

# Save model
os.makedirs('experiments/final', exist_ok=True)
with open('experiments/final/rf_booster.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Evaluate stand-alone RF
pred_rf = (p_rf > 0.5).astype(int)
print("\n--- RF Standalone Report ---")
print(classification_report(y_test, pred_rf))

# Load trust_pipeline.csv and fuse
trust_df = pd.read_csv('experiments/final/trust_pipeline.csv')

# Fuse predictions: we need to align indices (trust_df starts at index 11 of test_df)
p_rf_aligned = p_rf[11:]
y_test_aligned = y_test.iloc[11:].values

# Fused Probability: Max function for high recall
p_fused = [max(p_gnn, p_r) for p_gnn, p_r in zip(trust_df['p_attack'], p_rf_aligned)]
pred_fused = [1 if p > 0.5 else 0 for p in p_fused]

print("\n--- Fused Model (GNN + RF) Report ---")
print(classification_report(y_test_aligned, pred_fused))

tn, fp, fn, tp = confusion_matrix(y_test_aligned, pred_fused).ravel()
print(f"Fused TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

# Save fused probabilities to the pipeline
trust_df['p_rf'] = p_rf_aligned
trust_df['p_fused'] = p_fused
# We can overwrite p_attack so the dashboard automatically uses it, but we can also
# just update the dashboard to use p_fused. Let's just update p_attack directly 
# so the dashboard works out of the box!
trust_df['p_attack_gnn'] = trust_df['p_attack']
trust_df['p_attack'] = p_fused
trust_df.to_csv('experiments/final/trust_pipeline.csv', index=False)
print("\nOverwritten trust_pipeline.csv with fused probabilities.")
