import os
import json

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'exp_E2_results.json'), 'r') as f:
    res = json.load(f)

# Baseline & E1 metrics for comparison
# Virgin:
b_prec, b_rec, b_f1, b_pr, b_roc, b_thr = 0.9392, 0.9605, 0.9497, 0.5373, 0.8572, 0.86
# E1:
e1_prec, e1_rec, e1_f1, e1_pr, e1_roc, e1_thr = 0.9341, 0.9605, 0.9471, 0.6172, 0.8959, 0.98

# E2
e2_prec = res['validation_metrics']['Precision']
e2_rec = res['validation_metrics']['Recall']
e2_f1 = res['validation_metrics']['F1']
e2_pr = res['validation_metrics']['PR_AUC']
e2_roc = res['validation_metrics']['ROC_AUC']
e2_thr = res['validation_metrics']['selected_threshold']

# Determine if this supported the hypothesis better than E1
conclusion = "SUPPORT" if (e2_pr > e1_pr or e2_f1 > e1_f1) else "REJECT" if (e2_pr < e1_pr and e2_f1 < e1_f1) else "INCONCLUSIVE"

md = f"""# Experiment E2: Observation-Masked Max Pooling

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
| **Precision** | {b_prec:.4f} | {e1_prec:.4f} | {e2_prec:.4f} |
| **Recall** | {b_rec:.4f} | {e1_rec:.4f} | {e2_rec:.4f} |
| **F1 Score** | {b_f1:.4f} | {e1_f1:.4f} | {e2_f1:.4f} |
| **PR-AUC** | {b_pr:.4f} | {e1_pr:.4f} | {e2_pr:.4f} |
| **ROC-AUC** | {b_roc:.4f} | {e1_roc:.4f} | {e2_roc:.4f} |
| **Threshold** | {b_thr:.2f} | {e1_thr:.2f} | {e2_thr:.2f} |

## 3. Event-Level Validation Analysis (At E2 Threshold {e2_thr:.2f})
"""

for ev in res['event_level_validation']:
    md += f"- **Event {ev['event_id']} ({ev['duration_frames']} frames):** Max Prob = {ev['max_prob']:.4f}, Mean Prob = {ev['mean_prob']:.4f}, Detected: {ev['detected']}, Delay: {ev['detection_delay']} frames\n"

md += f"""
## 4. Conclusion
**Verdict:** {conclusion}

Interpretation: 
"""

with open(os.path.join(base_dir, 'experiments', 'experimental', 'gnn_gru_experiments', 'exp_E2_results.md'), 'w', encoding='utf-8') as f:
    f.write(md)
