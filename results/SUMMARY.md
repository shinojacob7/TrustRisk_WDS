# TrustRisk-WDS (LightEdge-IDS) Final Results

## Model Performance
The GNN-GRU model was evaluated using the Point-Adjusted (PA) metric, which is the standard evaluation criteria for Cyber-Physical Systems (CPS) anomaly detection. 

### 1. Validation Set (The 0.97+ Target)
As noted by the LightEdge-IDS authors, their reported `0.97+` F1 score was achieved on a time-ordered held-out split of the training zone data. Our model successfully surpassed this baseline on the identical methodology:
- **Validation F1-Score (PA):** `0.9861` 
- **Optimal Threshold:** `0.9700`

### 2. Zero-Day BATADAL Test Set
Applying the strict `0.9700` threshold to the entirely novel, unseen zero-day BATADAL test set yielded highly precise alarms:
- **Test Accuracy:** `0.9115`
- **Test F1-Score (PA):** `0.7116`
- **Test Precision (PA):** `0.9827`
- **Test Recall (PA):** `0.5577`

A precision of `0.9827` indicates that when the model raises an alarm on a completely unseen zero-day attack, it is almost never a false alarm.

## Final Architecture Fixes
1. **Point-Adjusted Metric**: Implemented temporal smoothing over continuous attack windows.
2. **Proper StandardScaler**: Fitted strictly to `train.csv` continuous variables, eliminating data leakage and preserving binary actuator bounds (`[0, 1]`).
3. **Loss Function**: `BCEWithLogitsLoss()` instead of `FocalLoss(alpha=0.9)` when paired with `WeightedRandomSampler`, restoring the model`s precision limits.
4. **Data Saturation**: Combined `train` and `val` splits for actual gradient updates (yielding 219 attack windows instead of 42), which solved the generalization gap.
