import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_normal = pd.read_csv('data/splits/train.csv')
train_attacks = pd.read_csv('data/splits/train_attacks.csv')
test_df = pd.read_csv('data/splits/test.csv')

# Clean train_attacks (remove -999 which means unknown)
train_attacks = train_attacks[train_attacks['ATT_FLAG'].isin([0, 1])]

train_full = pd.concat([train_normal, train_attacks])

features = [c for c in train_full.columns if c not in ['DATETIME', 'ATT_FLAG']]

X_train = train_full[features]
y_train = train_full['ATT_FLAG']

X_test = test_df[features]
y_test = test_df['ATT_FLAG']

rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

p_rf = rf.predict_proba(X_test)[:, 1]

# Fuse
trust_df = pd.read_csv('experiments/final/trust_pipeline.csv')
p_rf_aligned = p_rf[11:]

p_fused = [max(p_g, p_r) for p_g, p_r in zip(trust_df['p_attack_gnn'] if 'p_attack_gnn' in trust_df.columns else trust_df['p_attack'], p_rf_aligned)]
pred_fused = [1 if p > 0.5 else 0 for p in p_fused]

print(classification_report(y_test.iloc[11:], pred_fused))
tn, fp, fn, tp = confusion_matrix(y_test.iloc[11:], pred_fused).ravel()
print(f'TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}')

trust_df['p_attack'] = p_fused
trust_df.to_csv('experiments/final/trust_pipeline.csv', index=False)
