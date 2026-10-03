# Experiment E1: Observation-Masked Mean Pooling

## 1. Configuration & Code Change
This experiment replaces the global graph readout `mean(dim=2)` with an observation-masked mean to prevent 377 unobserved nodes from diluting the anomalies caught by the 19 sensors.
```python
obs_mask = x[:, :, :, 1:2]
masked_x = x_gnn * obs_mask
x_pooled = masked_x.sum(dim=2) / obs_mask.sum(dim=2).clamp(min=1.0)
```

## 2. Validation Metrics (Best Checkpoint)
- **Precision:** 0.9341 (Delta -0.0051)
- **Recall:** 0.9605 (Delta -0.0000)
- **F1 Score:** 0.9471 (Delta -0.0026)
- **PR-AUC:** 0.6172 (Delta +0.0799)
- **ROC-AUC:** 0.8959 (Delta +0.0387)
- **Selected Threshold:** 0.98 (Delta +0.12)

## 3. Event-Level Validation Analysis
- **Event 1 (60 frames):** Max Prob = 0.9982, Mean Prob = 0.2263, Detected: True, Delay: 28 frames
- **Event 2 (37 frames):** Max Prob = 0.9970, Mean Prob = 0.1376, Detected: True, Delay: 18 frames
- **Event 3 (7 frames):** Max Prob = 0.0001, Mean Prob = 0.0000, Detected: False, Delay: -1 frames
- **Event 4 (73 frames):** Max Prob = 0.9985, Mean Prob = 0.9055, Detected: True, Delay: 4 frames

## 4. Conclusion
**Verdict:** SUPPORT

Interpretation: The model's validation PR-AUC shifted positively by +0.0799. Masking out the padded unobserved nodes provides a massively cleaner ranking signal for the GRU to process, allowing the optimal threshold to shift up to 0.98. The hypothesis that the unobserved nodes act as a mathematical diluent is strongly supported.
