# Phase 6C: Physical Pipeline Diagnostic Audit

This report strictly diagnoses the physical data ingestion and graph-message passing architecture of the GNN-GRU pipeline. The code has been audited via runtime tensor inspection and static code analysis without any modifications.

## 1. Physical Feature Scaling & Magnitude Mismatch
**Status: CORRECT (No Leakage), but Zero-Diluted**
*   **SCADA Features (Measurement, Flow):** Correctly normalized using a `StandardScaler` perfectly restricted to `train_df`. Validation/test splits correctly use the frozen scaler.
*   **Physical Features (Elevation, Length, Diameter, Roughness):** Correctly normalized using static graph topology (`ctown_node_observation_mask.csv` & `ctown_edge_index.csv`). This topology is invariant to time, meaning there is zero data leakage.
*   **Binary Masks (Status, Observation Mask):** Correctly left unscaled as absolute `[0, 1]` tensors.
*   **Magnitude Check:** There is no runaway physical magnitude mismatch. `flow` peaks at 25.5 (extreme outliers, normal for pipes), but all other physical dimensions stay strictly near mean=0, std=1.
*   **Zero-Dilution Anomaly:** The std of `node_tensor[:,:,0]` (measurement) drops to `0.219`. This occurs because 377 of the 396 nodes are completely unobserved and safely zero-padded.

## 2. Observation Mask Handling
**Status: ARCHITECTURAL FLAW IDENTIFIED**
*   **Input Feature Only:** The mask is concatenated to the node features, but it **does not control message propagation or pooling**.
*   **Impact:** The 377 unobserved nodes pass `[0.0, 0.0, ...] ` vectors uniformly throughout the GNN layers and graph readouts.

## 3. Edge Feature Usage
**Status: CORRECT**
*   `edge_attr` is correctly concatenated directly into the message computation: `torch.cat([x_src, x_dst, edge_attr], dim=-1)`.
*   Ordering is deterministic and identically aligns with the PyG `edge_index`.

## 4. Graph Readout (The Critical Bottleneck)
**Status: CRITICAL SEVERITY (HIGH)**
*   **Mechanism:** `x_pooled = x_gnn.mean(dim=2)`
*   **The Issue:** The architecture performs a mathematically raw **Global Mean Pool across all 396 nodes**.
*   **Why this destroys detection (Events 1 & 6):** If a stealthy cyber-attack alters exactly 1 physical sensor to a massive anomaly value (e.g., embedding jump of `+5.0`), the mean pool divides this by 396 (`+0.0126`). To the downstream GRU, this microscopic fluctuation is entirely indistinguishable from random normal operational noise across the network.
*   **Conclusion:** The localized cyber-attack signals are being massively diluted by the 377 unobserved nodes.

## 5. Temporal Input
**Status: CORRECT**
*   Window size `W = 12` (12 hours) with `stride = 1`. 
*   It predicts the label strictly aligned to the final frame `t`. No future data leakage exists.

---

### 🚨 Final Verdict & Recommendation
The physical feature pipeline correctly handles scaling without leakage. However, the graph readout architecture (`mean` pooling) acts as a severe mathematical diluent against localized physical attacks. 

**Recommended Next Step:** Skip to **PHASE 6H (Experiment E: GNN Architecture / Oversmoothing)**. We must test `Max Pooling` or `Mean + Max Pooling` to allow localized attack spikes to survive the graph readout layer before hitting the GRU.
