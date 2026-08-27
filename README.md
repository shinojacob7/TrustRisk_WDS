# TrustRisk-WDS

**Trust-Aware Cyber-Physical Risk Intelligence and Critical Asset Protection Framework for Smart Water Distribution Systems**

TrustRisk-WDS is a comprehensive security framework designed for Smart Water Distribution Systems (SWDS). It goes beyond traditional anomaly detection by introducing a **Trust-Aware** validation mechanism. It calculates the structural vulnerability of the physical water network and combines it with dynamic AI trust metrics (Prediction Confidence, Epistemic Uncertainty, and Explanation Reliability) to help utility operators make reliable, informed decisions when an attack alert is generated.

This project represents the final, mathematically rigorous research implementation based on the **Seven-Phase Master Implementation Roadmap**.

## The 7-Phase Architecture

1. **Phase 1: Project Setup & Data Preprocessing:** Strict chronological sequence extraction of the BATADAL dataset, avoiding temporal leakage and preserving the natural time-series progression for train/val/test splits.
2. **Phase 2: Cyber-Physical Network & Physics Layer:** Implementation of formal physical risk engines based on C-Town EPANET topology:
   * **MDCI** (Asset Criticality via Structural Centrality)
   * **ASS** (Attack Susceptibility via Cyber-Exposure)
   * **ISS** (Impact Severity via Flow/Demand Metrics)
   * **CRI** (Cascading Risk via Hydraulic Reachability)
3. **Phase 3: AI-Based Intrusion Detection Layer:** Implementation of the **LightEdge-IDS** model—a Spatio-Temporal Graph Neural Network coupled with a Gated Recurrent Unit (GNN-GRU) built on a sensor-focused subgraph. (Also includes an unsupervised LSTM Autoencoder baseline).
4. **Phase 4: Trust & Reliability Evaluation Layer:**
   * **Probability Calibration:** LBFGS Negative Log-Likelihood optimization for Temperature Scaling.
   * **Uncertainty Quantification:** Monte Carlo (MC) Dropout for predictive variance and entropy.
   * **Explanation Reliability (ERS):** Formalized ERS using the Normalized Information Entropy of SHAP values.
5. **Phase 5: TrustRisk Intelligence Integration Layer:** Computes the Trust-Aware Dynamic Cyber Risk Index (TDCRI) and solves the Trust-aware Prioritized Protection Index (TPPI) using a **0/1 Knapsack Dynamic Programming Optimizer** for security resource allocation.
6. **Phase 6: Decision Support Dashboard:** An interactive Streamlit application deployed over the real-time test stream to visualize live alerts, WDS physical topology, and TPPI investment recommendations.
7. **Phase 7: Experimental Validation:** Strict adherence to data contracts preventing target leakage, utilizing properly formatted validation sets to evaluate ROC-AUC and F1 scores.

## Project Structure

```text
TrustRisk-WDS/
├── data/
│   ├── raw/                 # Raw BATADAL CSV datasets
│   ├── processed/           # Scaled datasets
│   └── splits/              # Strict chronological train/val/test splits
├── network/
│   ├── epanet/              # EPANET .INP files
│   ├── graph/               # C-Town NetworkX graph (.pkl)
│   └── mappings/            # BATADAL sensor to EPANET node mappings
├── preprocessing/           # Data cleaning and chronological sliding window generators
├── risk/                    # Formal mathematical Risk Engine (MDCI, ASS, ISS, CRI)
├── models/                  # AI Intrusion Detection Models
│   ├── baselines/           # Temporal BiLSTM Baseline
│   ├── lstm_autoencoder/    # Unsupervised Anomaly Detection
│   └── gnn_gru/             # LightEdge-IDS (GNN-GRU) Model
├── trust/                   # Calibration, MC Dropout, and SHAP ERS Evaluators
├── integration/             # TDCRI formulation and TPPI 0/1 Knapsack Solver
├── dashboard/               # Streamlit Real-Time Monitoring UI
├── experiments/             # Configs, run logs, and results
├── tests/                   # Unit tests
├── docs/                    # Project documentation and PDFs
└── requirements.txt         # Python dependencies
```

## Setup & Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

**1. Generate Chronological Data Splits (Phase 1)**
```bash
python preprocessing/build_dataset.py
```

**2. Train the Models (Phase 3)**
To train the LightEdge-IDS (GNN-GRU) model:
```bash
python models/gnn_gru/train_gnn_gru.py
```

**3. Launch the Interactive Decision Support Dashboard (Phase 6/7)**
To view the live alert simulation against the true chronological test dataset:
```bash
streamlit run dashboard/app.py
```
This will start a local server and automatically open the dashboard in your default web browser.
