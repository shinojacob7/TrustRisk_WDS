# Experiment E2: Observation-Masked Max Pooling

## 1. Configuration & Code Change
This experiment replaces the global graph readout `mean(dim=2)` with an observation-masked MAX pooling to ensure localized attack anomalies from the 19 observed nodes are not only undiluted but explicitly prioritized.
```python
obs_mask = x[:, :, :, 1:2] 
penalty = (1.0 - obs_mask) * 1e9
masked_x = x_gnn - penalty
x_pooled = masked_x.max(dim=2).values
```
*Note: We subtract 1e9 from unobserved nodes rather than multiplying by 0, ensuring that unobserved nodes cannot accidentally become the max if all legitimate activations are negative.*

## 2. Comparison: Virgin vs E1 vs E2

| Metric | Virgin Baseline | E1 (Masked Mean) | E2 (Masked Max) |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.9392 | 0.9341 | 0.9327 |
| **Recall** | 0.9605 | 0.9605 | 0.5480 |
| **F1 Score** | 0.9497 | 0.9471 | 0.6904 |
| **PR-AUC** | 0.5373 | 0.6172 | 0.2844 |
| **ROC-AUC** | 0.8572 | 0.8959 | 0.8102 |
| **Threshold** | 0.86 | 0.98 | 0.98 |

## 3. Event-Level Validation Analysis (At E2 Threshold 0.98)
- **Event 1 (60 frames):** Max Prob = 0.9953, Mean Prob = 0.1735, Detected: True, Delay: 49 frames
- **Event 2 (37 frames):** Max Prob = 0.9852, Mean Prob = 0.1140, Detected: True, Delay: 19 frames
- **Event 3 (7 frames):** Max Prob = 0.0000, Mean Prob = 0.0000, Detected: False, Delay: -1 frames
- **Event 4 (73 frames):** Max Prob = 0.0000, Mean Prob = 0.0000, Detected: False, Delay: -1 frames

## 4. Conclusion
**Verdict:** REJECT

Interpretation: 
