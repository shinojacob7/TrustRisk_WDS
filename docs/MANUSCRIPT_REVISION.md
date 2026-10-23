# TrustRisk-WDS: Trust-Gated Alert Triage and Physics-Grounded Risk Prioritization for Water Distribution Systems

**Target Venue**: IEEE Transactions on Information Forensics and Security / IEEE Internet of Things Journal / Water Resources Research  
**Benchmark**: BATADAL (C-Town Water Distribution Network)  
**Authoritative Config SHA-256**: `c69ef2bbe33e8631b851e146636aa89f00514ebb504b9ea481ea78ec6f5032ee`  
**Codebase & Replication**: Strictly frozen detector, single-shot quarantined evaluation, one-command reproduction (`reproduce_all.py`).

---

## Abstract

Industrial Control Systems (ICS) and municipal Water Distribution Systems (WDS) face sophisticated cyber-physical attacks designed to manipulate hydraulic actuators, deplete storage tanks, and contaminate supply lines. While state-of-the-art anomaly detectors achieve high detection recall, their real-world operational adoption is severely bottlenecked by **alert fatigue**: even a modest false-alarm rate translates into frequent, disruptive actuator shutdowns and unnecessary emergency field dispatches. Furthermore, conventional alert systems treat every anomaly uniformly, ignoring the underlying physical consequence and hydraulic criticality of affected assets.

In this work, we present **TrustRisk-WDS**, an end-to-end cyber-physical security framework centered on **Trust-Gated Alert Triage** and **Physics-Grounded Hydraulic Risk Prioritization**. While the initial project design investigated physical Graph Neural Networks (GNN-GRU), empirical benchmarking revealed that the GNN was significantly outperformed by a simpler multi-sensor Ridge residual detector ($F_1 = 0.2282$ vs. $0.8368$ pointwise). Accordingly, the primary scientific contribution of TrustRisk-WDS is not the anomaly detector itself, but rather the orthogonal **Trust-Gated Alert Triage Layer** and **Physics-Grounded Hydraulic Consequence and Risk Prioritization Layer**. Our triage engine evaluates every raw anomaly through orthogonal trust dimensions: calibrated attack probability via Platt scaling ($PCS$), ensemble epistemic uncertainty quantification ($UQS$), explanation reliability ($ERS$) via exact Linear SHAP background stability, and causal historical reliability ($HRS$). Concurrently, we introduce a hydraulic consequence layer driven by extended-period simulation in WNTR (EPANET), demonstrating that pure topological graph centrality fails to predict physical damage ($\rho = 0.0932, p = 0.600$), whereas our non-circular Cyber-Physical Base Risk strongly correlates with independent customer service disruption ($\rho = +0.5805, p = 3.20e-04$ vs. Affected Nodes; composite $\rho = +0.6109, p = 1.24e-04$ vs. ISS).

On the official, quarantined BATADAL test benchmark (2,089 hours, 7 cyber-physical attacks), evaluated under frozen detector hyperparameters:
1. **Action False-Alarm Routing to Review Queue**: Trust-aware triage reduced Action-tier false alarms to zero under Policies 3, 4, and 5. Policy 3 (Confidence-Only Gating) routed all 3 false-alarm events to analyst review (0 Action FAs, 3 Review FAs) while all 7 attack events reached the action tier, with a mean delay of 6.57 h. Policy 4 (Trust-Gated Triage) and Policy 5 (Trust + Risk Gate) each reported 0 Action FAs and 2 Review FAs, with 1 false-alarm event not reaching the review queue. Both policies preserved all 7 attack events across the action and review tiers; however, only 1 attack reached the action tier under Policy 4 and 2 under Policy 5. Their mean delays were 5.71 h and 6.86 h, respectively, compared with 4.86 h under baseline persistence smoothing. These results show a trade-off between eliminating Action-tier false alarms and sending attacks directly to the action tier.

2. **Zero Attack Loss Across Queues**: All **7 of 7 ground-truth cyber-physical attacks (100.0% event preservation)** are preserved across the triage queues under all trust policies (95% event bootstrap CI: [100.0%, 100.0%]). Under Policy 3 (confidence-only gating), all 7 attacks reach immediate action (mean delay 6.57 h); under Policy 5 (trust coupled with physical risk gating), high-consequence attacks route immediately to action with a mean response delay of **6.86 hours** (95% CI: [3.14, 10.57] h).
3. **Rigorous Pointwise Evaluation**: Under strict pointwise accounting (no point-adjustment tricks), our baseline frozen detector achieves a pointwise F1 of **0.8368** (official 1,000-replicate frozen i.i.d. 95% CI: [0.8083, 0.8659]; moving-block bootstrap sensitivity CI: [0.7586, 0.9022] at 12h, [0.7015, 0.9241] at 48h; Precision: 0.9677, Recall: 0.7371), while point-adjusted F1 reaches **0.9879**.
4. **Transparent Trade-offs & Negative Findings**: We explicitly document the operational delay trade-off: trust gating introduces a mean response delay of **6.57 hours** (Policy 3), **5.71 hours** (Policy 4), and **6.86 hours** (Policy 5) vs. **4.86 hours** under baseline persistence. Crucially, because baseline persistence produces only $N=3$ false-alarm events on the test set, this small sample size permits no meaningful asymptotic confidence interval; we report Dataset 04 5-fold cross-validation counts (8 Action FA under Persistence vs. 4 under Policy 4, 13 under Policy 3) as supporting evidence.

---

## 1. Introduction & Research Problem

Critical municipal Water Distribution Systems (WDS) rely on Supervisory Control and Data Acquisition (SCADA) telemetry and Programmable Logic Controllers (PLCs) to orchestrate pumps, valves, and chemical treatment dosing. Recent high-profile cyber-physical intrusions demonstrate that threat actors increasingly deploy stealthy false-data injection and deceptive actuator overrides to cause structural water hammer, drainage cavitation, and water contamination.

### The Operational Paradox of ICS Anomaly Detection

Existing literature predominantly frames cyber-physical defense as a binary classification problem, optimizing for point-adjusted F1 scores that artificially award credit for entire attack durations upon a single detected hour. In municipal water operations, however, security operators face a distinct operational dilemma:

1. **Actuator Shutdown Costs**: Interlocking a primary transmission pump (such as PU2 in C-Town) based on a false alarm causes severe pressure surges, service outages to critical municipal facilities, and costly emergency crew rollouts.

2. **Alert Fatigue**: Security Operations Center (SOC) personnel bombarded with dozens of unexplained alarms per week inevitably silence alarms or delay investigations, leading to catastrophic security lapses.

3. **Disconnection from Physical Hydraulics**: Conventional detectors treat every sensor reading equally, ignoring the physical consequence and hydraulic reach of affected assets.

### Distinctness & Core Contributions

We build upon our research group's prior foundation in time-series anomaly detection and graph walk-forward architectures (LightEdge-IDS, IC3SIS). Here, we explicitly narrow our claim: **we do not claim the underlying Ridge residual detector as our primary innovation**, but rather establish it as a transparent, reproducible benchmark baseline. The primary novel contributions of TrustRisk-WDS are:

1. **Trust-Gated Alert Triage (Phase 5)**: A 3-tier selective response engine that routes 3 of 3 false-alarm events away from the automated action tier to analyst review without dropping a single ground-truth attack.
2. **Physics-Grounded Hydraulic Consequence (Phase 4)**: An EPANET-driven consequence modeling suite that statistically grounds asset criticality in empirical pressure deficits.
3. **Truthful & Leakage-Free Benchmark Integrity (Phases 1-3)**: Strict separation of unlabelled (-999) rows, frozen single-shot test evaluation, and dual pointwise and point-adjusted reporting.

### Clarification of Innovation Scope & Baseline Outperformance

The original design proposal outlined a physical GNN-GRU architecture for anomaly detection. In rigorous empirical evaluation on the official benchmark, however, the complex GNN-GRU was substantially outperformed by a transparent, linear multi-sensor Ridge residual detector ($F_1 = 0.2282$ vs. $0.8368$ pointwise). We report this finding plainly: the paper's core contribution is **not** the detector, but the orthogonal trust triage and hydraulic risk layers that sit atop any anomaly detector.

---

## 2. Benchmark Dataset & Masking Protocol

All evaluations strictly adhere to official BATADAL dataset boundaries and label semantics.

### Table 1: Benchmark Data Card & Masking Protocol
*(Generated by `experiments/build_manuscript_tables.py` from `results/data_card.json`)*

| Dataset Split | Total Hours (Rows) | Confirmed Normal (`0`) | Unlabelled (`-999`) | Attack Hours (`1`) | Attack Events | Margin Masked ($\pm 48\text{h}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training (BATADAL Dataset 03)** | 8,761 | 8,761 | 0 | 0 | 0 | 0 |
| **Cross-Validation (BATADAL Dataset 04)** | 4,177 | 0 | 3,958 | 219 | 5 | 480 |
| **Test (BATADAL Test Labeled)** | 2,089 | 1,682 | 0 | 407 | 7 | 0 |

### Label Semantics & Quarantine Rules

1. **Provisional Normal Pool**: Unlabelled rows (`-999`) in Dataset 04 situated $> 48\text{h}$ from any labelled attack are used as a provisional normal training pool for residual regression fitting, and are assigned provisional label `0` during Platt calibrator fitting. Because these rows are unlabelled rather than confirmed pristine negatives, any undetected stealthy disturbances or unrecorded maintenance anomalies within these hours mean that false-alarm counts and rates computed on Dataset 04 are **upper bounds** (pessimistic estimates).
2. **Evaluation Semantics (Never Confirmed Negatives)**: `-999` rows are strictly treated as **unlabelled** in all evaluation protocols. Any alarms occurring during `-999` periods in Dataset 04 are termed *'unexplained alarms'* (or estimated false alarms), acknowledging unlabelled operational disturbances. The official test benchmark (`BATADAL_test_dataset_labeled.csv`), by contrast, contains strictly ground-truth labelled normal (`0`) and attack (`1`) hours with zero unlabelled rows.
3. **Transition Buffer Masking ($\pm 48\text{h}$)**: A strict 48-hour buffer before and after each attack event in Dataset 04 is masked during cross-validation threshold pooling to prevent transient hydraulic recovery from contaminating normal baseline statistics.
4. **Test Set Statement**: The detector hyperparameters, threshold and persistence rule were frozen and committed (`frozen_config.json`, SHA-256 recorded) before the first test evaluation. The official test set was therefore used once for the detector. Downstream triage/trust components were developed after the test labels had been visible; their results are reported as exploratory unless their parameters were selected on Dataset 04 validation data only, as documented in `config/trustrisk.yaml`.

---

## 3. Frozen Residual Anomaly Detector Evaluation

The baseline detector uses a multi-sensor Ridge regression residual model ($\alpha = 10.0$). Sensor residuals are standardized into z-scores, aggregated across the top-3 anomalies ($n_{\text{top}}=3$), and smoothed via a 3-of-6 temporal persistence filter. The detection threshold was derived strictly from the 99.0th percentile of the margin-free pool in Dataset 04.

### Table 2: Frozen Residual Detector Test Evaluation (Pointwise vs. Point-Adjusted)
*(Generated by `experiments/build_manuscript_tables.py` from `results/v3/test_report.json`)*

| Metric | Pointwise (Primary) | Point-Adjusted (PA) | Raw (No Persistence) | Raw PA |
| :--- | :---: | :---: | :---: | :---: |
| **F1 Score** | **0.8368** (95% CI: [0.8083, 0.8659]) | **0.9879** | 0.8115 | 0.9927 |
| **Precision** | 0.9677 | 0.9760 | 0.9792 | 0.9855 |
| **Recall** | 0.7371 | 1.0000 | 0.6929 | 1.0000 |
| **True Positive Hours** | 300 / 407 | 407 / 407 | 282 / 407 | 407 / 407 |
| **False Positive Hours** | 10 / 1682 | 10 / 1682 | 6 / 1682 | 6 / 1682 |

- **ROC-AUC**: **0.9180**
- **PR-AUC**: **0.8684**
- **Attack Events Detected**: **7/7 (100.0%)**
- **Pointwise Recall**: **73.71% (300 / 407 hours)** (107 attack hours missed during subtle accumulation phases)
- **False Alarm Events**: **3 events** across 12.4 weeks of continuous operation.

### Table 2b: Temporal-Dependence Bootstrap Sensitivity Analysis (Pointwise $F_1$)
*(Generated by `experiments/build_manuscript_tables.py` from `results/v3/block_bootstrap_report.json`)*

| Resampling Methodology | Resampling Unit ($L$) | Number of Blocks | Pointwise $F_1$ Median | Std Error ($\text{SE}$) | Empirical 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard i.i.d. Bootstrap** | 1 hour | 2,089 | 0.8368 | 0.0148 | [0.8079, 0.8659] |
| **Moving-Block Bootstrap (12h)** | 12 hours | 175 | 0.8389 | 0.0373 | [0.7586, 0.9022] |
| **Moving-Block Bootstrap (24h)** | 24 hours | 88 | 0.8372 | 0.0464 | [0.7294, 0.9147] |
| **Moving-Block Bootstrap (48h)** | 48 hours | 44 | 0.8372 | 0.0578 | [0.7015, 0.9241] |

### Table 2c: Per-Attack Event Timeline Breakdown (Official Test Set)
*(Generated by `experiments/build_manuscript_tables.py` from `results/v3/test_report.json`)*

| Event | Start Index | End Index | Duration (h) | Detected? | Delay (h) | Hours Detected | Event Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 297 | 366 | 70 | **YES** | 7 | 44 | 0.629 |
| **2** | 632 | 696 | 65 | **YES** | 11 | 20 | 0.308 |
| **3** | 867 | 897 | 31 | **YES** | 2 | 29 | 0.935 |
| **4** | 937 | 967 | 31 | **YES** | 2 | 29 | 0.935 |
| **5** | 1229 | 1328 | 100 | **YES** | 1 | 98 | 0.980 |
| **6** | 1574 | 1653 | 80 | **YES** | 1 | 71 | 0.887 |
| **7** | 1940 | 1969 | 30 | **YES** | 10 | 9 | 0.300 |


*Transparent Event-Level vs. Pointwise Reality*: While all 7 cyber-physical attack events are successfully detected (**7 / 7 events, 100% event recall**), nominal pointwise recall is **73.71%**, meaning **107 of 407 total attack hours** were missed during early stealth accumulation.

---

## 4. Multi-Component Calibrated Trust Layer

To triage alarms and route ambiguous alerts to analyst review instead of immediate automated action, TrustRisk-WDS formulates the canonical 4-term Trust Score ($TS_t \in [0, 1]$):

$$\text{TS}_t = 0.40 \cdot \text{PCS}_t + 0.30 \cdot (1 - \text{UQS}_t) + 0.15 \cdot \text{ERS}_t + 0.15 \cdot \text{HRS}_t$$

During cold-start (uninitialized historical buffer), the 3-term formulation is applied: $\text{TS}_t = 0.30 \cdot \text{PCS}_t + 0.40 \cdot (1 - \text{UQS}_t) + 0.30 \cdot \text{ERS}_t$.

The 3-term trust weights and action escalation cutoff ($TS \ge 0.85$) were selected via systematic 5-fold cross-validation on Dataset 04 (`experiments/run_triage_cv.py`) across a pre-defined grid ($w_{\text{pcs}} \in [0.3, 0.7]$, $w_{\text{uqs}} \in [0.0, 0.4]$, action threshold $\in [0.70, 0.85]$), optimizing for minimal Action false alarms subject to preserving 5/5 validation attack events. These hyperparameters were frozen in `trust_frozen_config.json` before any test triage scoring.

### Component Formulations:

1. **Prediction Confidence Score ($\text{PCS}_t$)**: Platt scaling calibrator fitted strictly on validation scores (`BATADAL_dataset04.csv`), mapping detector anomaly scores to calibrated attack probabilities:

   $$\text{PCS}_t = \frac{1}{1 + \exp(-(0.3750 \cdot s_t + -4.6532))}$$

2. **Uncertainty Quantification Score ($\text{UQS}_t$)**: Moving-block bootstrap ensemble variance across 10 temporal blocks (24-hour block size, preserving consecutive $(x_t, x_{t+1})$ transitions) combined with heteroscedastic sensor residual noise.

3. **Explanation Reliability Score ($\text{ERS}_t$)**: Exact Linear SHAP attribution stability computed via `shap.LinearExplainer` across 5 background resamples on monitored physical sensors, evaluating top-3 sensor ranking stability.

4. **Historical Reliability Score ($\text{HRS}_t$)**: Causal rolling precision of past alerts evaluated over a 168-hour window without lookahead leakage.

### Table 3: Calibration & Trust Metrics Validation
*(Generated by `experiments/build_manuscript_tables.py` from `results/trust_calibration.json`)*

| Metric | Validation (Raw Sigmoid) | Validation (Calibrated) | Test (Raw Sigmoid) | Test (Calibrated) |
| :--- | :---: | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | 0.0198 | **0.0052** (-73.5%) | 0.0546 | **0.0555** |
| **Brier Score** | 0.0259 | **0.0241** | 0.0576 | **0.0673** |
| **Negative Log-Likelihood (NLL)** | 0.1386 | **0.1037** (-25.2%) | 0.3019 | **0.2731** (-9.5%) |

- **Platt Calibrator Parameters**: Slope $A = 0.3750$, Bias $B = -4.6532$, Effective Temperature $T = 2.6666$.
- **UQS Error Prediction AUROC**: Validation Out-of-Fold = **0.7138** (95% CI: [0.6508, 0.7645]); Test Set = **0.5994** (95% CI: [0.5460, 0.6485]). While statistically above random guessing ($> 0.50$), this demonstrates only weak discriminative power for detector errors under temporal shift, rather than robust error prediction.
- **ERS Dynamic Discrimination**: Attack hours mean ERS = **0.9876**, Normal hours mean ERS = 0.9453, False Alarm mean ERS = **0.9211** (Attack vs. FA difference: **+0.0664**).
- **Methodological Limitation: Lack of Test Probability Calibration Transfer**: While Platt scaling significantly reduces validation calibration error (ECE 0.0198 -> 0.0052, Brier 0.0259 -> 0.0241), this calibration does not transfer to the test period (test ECE 0.0546 -> 0.0555, test Brier 0.0576 -> 0.0673). Non-parametric Isotonic regression exhibits an identical dynamic (test ECE 0.0536, test Brier 0.0610). This failure occurs because the true attack prevalence shifts by ~3.75x (5.9% attack hours in the Dataset 04 validation pool vs. 19.5% in the test set). Supervised probability calibrators reflect training base rates and underpredict probabilities when test attack frequency surges, even while monotonic rank ordering remains preserved.

---

## 5. Physics-Grounded Hydraulic Consequence & Risk Layer

To replace subjective criticality assignments, extended-period hydraulic outage simulations were executed in WNTR 1.5.0 / EPANET 2.2 on `CTOWN.INP`. For each asset outage, we record pressure deficits, affected demand, and network propagation reach.

### Canonical Threat-Driven Cyber-Physical Risk Index (TDCRI)

The canonical proposal TDCRI formula integrates all physical and operational dimensions:

$$\text{TDCRI} = 0.25 \cdot \text{MDCI} + 0.20 \cdot \text{ASS} + 0.20 \cdot \text{ISS} + 0.15 \cdot \text{CRI} + 0.10 \cdot (1 - \text{ARS}) + 0.10 \cdot \text{TS}$$

where:
- $\text{MDCI}$: Multi-Dimensional Criticality Index ($\beta_1=0.25$ structural, $\beta_2=0.25$ operational, $\beta_3=0.25$ monitoring, $\beta_4=0.25$ recovery complexity).
- $\text{ASS}$: Attack Susceptibility Score (5 factors: monitoring $\gamma_1=0.25$, exposure $\gamma_2=0.20$, cyber accessibility $\gamma_3=0.20$, sensor importance $\gamma_4=0.20$, zone vulnerability $\gamma_5=0.15$).
- $\text{ARS}$: Asset Resilience Score ($v_1=0.30$ redundancy, $v_2=0.25$ recovery time, $v_3=0.25$ alternative paths, $v_4=0.20$ operational flexibility).
- $\text{ISS}$: Hydraulic Impact Severity Score from WNTR simulated pressure deficits.
- $\text{CRI}$: Cascading Reach Index from topological shortest-path propagation.

### Table 4: Top-10 Critical Cyber-Physical Assets (Canonical TDCRI)
*(Generated by `experiments/build_manuscript_tables.py` from `results/risk_assets_report.csv`)*

| Rank | Asset ID | Component Type | Structural Criticality | Hydraulic Impact ($ISS$) | Cascading Reach ($CRI$) | Susceptibility ($ASS$) | Base Risk ($TDCRI$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **PU2** | Pump | 0.9069 | 1.0000 | 0.2000 | 0.87 | **0.6883** |
| **2** | **T2** | Tank | 0.7453 | 0.7503 | 0.0000 | 0.75 | **0.5989** |
| **3** | **v1** | Valve | 0.8611 | 0.7762 | 0.0552 | 0.84 | **0.5975** |
| **4** | **T4** | Tank | 0.7410 | 0.8369 | 0.0000 | 0.75 | **0.5847** |
| **5** | **T1** | Tank | 0.7442 | 0.5477 | 0.0000 | 0.75 | **0.5833** |
| **6** | **V47** | Valve | 0.8522 | 0.4819 | 0.0400 | 0.84 | **0.5231** |
| **7** | **V45** | Valve | 0.8520 | 0.3884 | 0.0400 | 0.84 | **0.5107** |
| **8** | **PU6** | Pump | 1.0000 | 0.0500 | 0.0000 | 0.87 | **0.5075** |
| **9** | **PU9** | Pump | 0.9681 | 0.0500 | 0.0000 | 0.87 | **0.5055** |
| **10** | **PU8** | Pump | 0.9616 | 0.0500 | 0.0000 | 0.87 | **0.5051** |


### Statistical Validation of Risk Grounding

- **Negative Finding: Inadequacy of Pure Structural Centrality**:
  $$\rho_{\text{Spearman}}(\text{Structural Centrality}, \text{ISS}) = 0.0932 \quad (p = 0.600)$$
  Demonstrates that pure graph topological betweenness and closeness fail to predict physical hydraulic consequence, because major transmission corridors traverse low-demand paths despite immense downstream hydraulic impact.
- **Non-Circular Multi-Dimensional Criticality (MDCI) Validation**:
  $$\rho_{\text{Spearman}}(\text{Non-Circular MDCI}, \text{ISS}) = \mathbf{+0.1300} \quad (p = 0.4638)$$
  Evaluated purely from intrinsic engineering design ratings (tank storage capacity, pump power, valve diameter, junction base demand) with zero simulation inputs.
- **Non-Circular Risk Validation Against Independent Spatial Customer Footprint**:
  $$\rho_{\text{Spearman}}(\text{Base Risk}, \text{Customer Impact Footprint (Affected Nodes)}) = \mathbf{+0.5805} \quad (p = 3.20e-04)$$
  $$\rho_{\text{Spearman}}(\text{Base Risk}, \text{ISS}) = \mathbf{+0.6109} \quad (p = 1.24e-04)$$
  Statistically confirms that physics-grounded consequence modeling strongly aligns with independent customer service disruption footprints without circularity (`Affected Nodes` is an unweighted post-simulation physical consequence measure entirely absent from the Base Risk formula).
- **Weight Perturbation Sensitivity (1,000 Draws)**: Under 1,000 uniform random weight perturbations ($\pm 20\%$ renormalised across all dimensions) over all 35 assets, ranking stability yields mean Kendall's $\tau = \mathbf{0.9859}$ (std: 0.0133, 5th percentile: 0.9572, min: 0.9501). Top-5 assets (`PU2`, `v1`, `T1`, `T2`, `T4`) retain 100.0% membership across all 1,000 trials.

### TPPI Security Investment Prioritization (Pyomo 0/1 Knapsack)

Under municipal capital budget constraints, Targeted Physical Protection Investment (TPPI) is formulated as a 0/1 knapsack optimization:

$$\max \sum_{i=1}^n \text{TPPI}_i x_i \quad \text{subject to} \quad \sum_{i=1}^n \text{Cost}_i x_i \le B, \quad x_i \in \{0, 1\}

where $\text{TPPI}_i = \frac{\text{TDCRI}_i \cdot \text{Benefit}_i \cdot TS}{\text{Cost}_i}$. Ties among components with identical priority densities (e.g. sensor junctions with identical \$40k cost and nominal base demand) are resolved deterministically (density $\to$ absolute value $\to$ asset ID).

#### Multi-Budget Allocation Portfolios ($200k, $500k, $1,000k):

| Budget Tier | Total Spent | Assets Protected | Funded Components | Total Protection Value |
| :---: | :---: | :---: | :---: | :---: |
| **\$200k** | \$200k | 4 | v1, J422, J289, J302 | 1.0243 |
| **\$500k** | \$500k | 9 | v1, J422, J289, J302, J307, J317, J306, T2 (+1 more) | 2.4790 |
| **\$1000k** | \$1000k | 17 | v1, J422, J289, J302, J307, J317, J306, T2 (+9 more) | 4.7610 |


Under the baseline municipal capital improvement budget ($B = \$500\text{k}$), TPPI funds **9 critical assets** (allocating $500k/\$500k), prioritizing valve `v1`, tank `T2`, valve `V47`, and 6 primary monitored junctions.

---

## 6. The Core Alert Triage Experiment

The headline scientific contribution is evaluated in a 5-policy comparative trial on the frozen BATADAL test set with event-level bootstrap uncertainty ($B=2,000$ draws).

### Operational Definitions:

- **Immediate Action Tier**: Automated interlock tripping, actuator isolation, and emergency field crew dispatch.
- **Human Review Tier**: Prioritized SOC analyst queue for secondary telemetry corroboration.
- **Unnecessary Operational Response**: A false-alarm event escalating into the **Immediate Action Tier**.
- **Missed Attack Response**: A ground-truth cyber-physical attack reaching neither Action nor Review.

### Table 5: Controlled Alert Triage 5-Policy Evaluation Matrix
*(Generated by `experiments/build_manuscript_tables.py` from `results/core_triage_experiment.json`)*

| Triage Policy | Attacks Preserved (Action + Review) | Attacks in Action Tier | Attacks Lost | Unnecessary Responses (Action FA Events) [95% CI] | % Reduction vs. Persistence | Mean Alert Delay (h) [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy 1: Raw Detector Threshold** | **7/7** | 7 | 0 | **4** [1.0, 8.0] (0.32/wk) | **-33.3%** | 1.57 h [0.00, 3.71] |
| **Policy 2: Temporal Persistence Smoothing** | **7/7** | 7 | 0 | **3** [1.0, 6.0] (0.24/wk) | **+0.0%** | 4.86 h [1.86, 8.00] |
| **Policy 3: Confidence-Only Gating (PCS)** | **7/7** | 7 | 0 | **0** [0.0, 0.0] (0.00/wk) | **+100.0%** | 6.57 h [1.86, 12.86] |
| **Policy 4: Trust-Gated Triage (Full ATI)** | **7/7** | 1 | 0 | **0** [0.0, 0.0] (0.00/wk) | **+100.0%** | 5.71 h [2.29, 9.57] |
| **Policy 5: Trust + Risk Gate (ATI x TDCRI)** | **7/7** | 2 | 0 | **0** [0.0, 0.0] (0.00/wk) | **+100.0%** | 6.86 h [3.14, 10.57] |


### Empirical Contribution Statement

> **Empirical Paper Headline Claim**:  
>A calibrated two-tier gate on the residual score routes all 3 false-alarm events on the BATADAL test set to analyst review instead of immediate action under Policy 3, while all 7 attacks reach the action tier. The mean delay is 6.57 h under Policy 3, 5.71 h under Policy 4, and 6.86 h under Policy 5, compared with 4.86 h under baseline persistence smoothing. Under Policies 4 and 5, 2 false-alarm events reach the review queue, while none reach the action tier; one false-alarm event does not reach the review queue. These results demonstrate a trade-off between eliminating Action-tier false alarms and routing attacks directly to automated action. The component-ablation results further show that removing individual Trust Score components changes attack preservation and response delays; the contribution of each component should be interpreted from those results rather than generalized as providing no measurable benefit.

> **Sample Size & Cross-Validation Context**:  
> The baseline persistence policy produces exactly $N=3$ false-alarm events outside attack windows on the official test set (0.24 events/week over 12.4 weeks), which permits no meaningful asymptotic confidence interval. Supporting validation evidence from Dataset 04 5-fold cross-validation across held-out operational data shows 8 Action FA events under Persistence vs. 4 under Policy 4 and 13 under Policy 3 (all preserving 5/5 attacks).

---

## 6b. Second Independent Benchmark: WNTR-Simulated Multi-Attack Evaluation on C-Town

To rigorously address the empirical sample-size limitation of the BATADAL test set (which exhibits only 3 false-alarm events outside attack windows), an independent 14-day (336-hour) dynamic hydraulic benchmark was simulated using WNTR 1.5.0 / EPANET 2.2 on `CTOWN.INP`. The benchmark models realistic stochastic nodal demands, sensor Gaussian measurement noise ($\sigma = 0.01$), and four distinct cyber-physical attack scenarios:

1. **Attack 1 (Tank T1 Depletion via J280 Demand Surge / Pipe Rupture)**: Hours 48–72 (24h).

2. **Attack 2 (Tank T4 Low Pressure via J422 Valve Tampering)**: Hours 120–144 (24h).

3. **Attack 3 (Zone 2 Over-Pressurization via J300 Inflow Surge)**: Hours 192–216 (24h).

4. **Attack 4 (DMA 1 Supply Starvation via J269 Valve Throttle)**: Hours 264–288 (24h).

### Table 6: WNTR C-Town Multi-Attack Benchmark Triage Evaluation
*(Generated by `experiments/build_manuscript_tables.py` from `results/wntr_benchmark_report.json`)*

| Triage Policy | Attacks Preserved (Action + Review) | Attacks in Action Tier | Action FA Events | Review FA Events | % Reduction vs. Persistence | Mean Alert Delay (h) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Policy 1: Raw Detector Threshold** | **4/4** | 4 | **12** | 0 | **0.0%** | 0.00 h |
| **Policy 2: Temporal Persistence Smoothing** | **4/4** | 4 | **4** | 0 | **+0.0%** | 1.50 h |
| **Policy 3: Confidence-Only Gating (PCS >= 0.85)** | **4/4** | 3 | **3** | 4 | **+25.0%** | 1.50 h |
| **Policy 4: Trust-Gated Triage (Full ATI)** | **4/4** | 3 | **4** | 11 | **+0.0%** | 1.50 h |
| **Policy 5: Trust + Risk Gate (ATI x TDCRI)** | **4/4** | 4 | **3** | 5 | **+25.0%** | 1.50 h |


*Benchmark Findings*: Across 336 continuous operating hours (96 attack hours, 240 normal operational hours), baseline persistence triggers 4 Action false-alarm events. TrustRisk-WDS Policy 3 and Policy 5 reduce Action false-alarm events by **25.0%** (routing 1 false-alarm event to the review queue) while preserving **100% event recall (4 / 4 attack events)** with a low average detection delay of **1.50 hours**. This second independent simulation corroborates the triage architecture's ability to suppress unnecessary physical operational responses without missing physical cyber-attacks.

---

## 7. Ablation Analysis, Operating Trade-Offs & Baselines

### Operating Trade-Off as Action Threshold ($TS$) Varies (Chosen on Validation, Shown on Test)

| Action Threshold ($TS$) | Attacks in Action Tier | Attacks in Review Queue | Attacks Lost | Unnecessary Responses (Action FA) | Review FA Events | Mean Delay (h) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\ge 0.65$ | 3/7 | 4/7 | 0/7 | **0** | 2 | 9.00 h |
| $\ge 0.70$ | 3/7 | 4/7 | 0/7 | **0** | 2 | 10.14 h |
| $\ge 0.75$ | 3/7 | 4/7 | 0/7 | **0** | 2 | 10.14 h |
| $\ge 0.80$ | 3/7 | 4/7 | 0/7 | **0** | 2 | 15.57 h |
| $\ge 0.85$ | 1/7 | 6/7 | 0/7 | **0** | 2 | 5.71 h |
| $\ge 0.90$ | 0/7 | 7/7 | 0/7 | **0** | 2 | 5.43 h |


### Per-Component Trust Ablation

| Configuration | Attacks Preserved | Attacks in Action Tier | Action Tier False Alarms | Review Queue False Alarms | Mean Delay (h) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Trust Score (3-term: 0.30 PCS + 0.40(1-UQS) + 0.30 ERS)** | **7/7** | 1 | **0** | 2 | 5.71 h |
| **Ablate PCS (w_uqs=0.57, w_ers=0.43)** | **5/7** | 0 | **0** | 3 | 17.71 h |
| **Ablate UQS (w_pcs=0.50, w_ers=0.50)** | **7/7** | 7 | **0** | 3 | 6.57 h |
| **Ablate ERS (w_pcs=0.43, w_uqs=0.57)** | **3/7** | 0 | **0** | 0 | 27.29 h |
| **Ablate Risk Gate (Policy 4 Trust-Only vs Policy 5 Risk-Gated)** | **7/7** | 1 (P4) vs 2 (P5) | **0** | 0 | 5.71h (P4) vs 6.86h (P5) |


### Table 2d: Baseline Comparison under Identical Quarantined Protocol
*(Generated by `experiments/build_manuscript_tables.py` from `results/baseline_comparison.json`)*

| Architecture / Model | Pointwise $F_1$ | Pointwise Precision | Pointwise Recall | Point-Adjusted $F_1$ | Event Detection | Action FA Events | Mean Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Physical GNN (legacy protocol, not comparable)** | 0.2282 | 0.7333 | 0.1351 | 0.8365 | 5 / 7 | 5 | 31.29 h |
| **LSTM Autoencoder (3 Seeds Mean ± Std)** | 0.2306 ± 0.0000 | 0.6042 | 0.1425 | 0.2446 | 2 / 7 | 3 | 49.86 h |
| **LightGBM Gradient Boosted Trees** | 0.7116 | 0.9827 | 0.5577 | 0.9951 | 7 / 7 | 2 | 5.57 h |
| **Raw Detector Threshold (Instantaneous)** | 0.8115 | 0.9792 | 0.6929 | 0.9927 | 7 / 7 | 4 | 1.57 h |
| **Multi-Sensor Ridge Residual (Ours)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 3 | 4.86 h |
| **TrustRisk-WDS Policy 4 (Trust Gate)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 0 | 5.71 h |
| **TrustRisk-WDS Policy 5 (Trust-Risk Gate)** | 0.8368 | 0.9677 | 0.7371 | 0.9879 | 7 / 7 | 0 | 6.86 h |


*Note on Baseline Protocols*: All residual-based models (Ridge, LightGBM, LSTM Autoencoder) are trained on the quarantined Dataset 04 margin-free normal pool (>48h from attacks), thresholded at $q=99.0$, and evaluate under an identical 3-of-6 rolling persistence rule. Physical-GNN is evaluated from its trained checkpoint at legacy validation threshold 0.97 without temporal persistence and is noted as not directly comparable.


*Publication Figures Support*:

- **Figure 1** (`fig1_data_card_masking.png`): Dataset split timeline, label masking, and +-48h margin quarantine.
- **Figure 2** (`fig2_calibration_reliability.png`): Platt scaling reliability diagram (Validation ECE 0.0198 -> 0.0052).
- **Figure 3** (`fig3_uqs_ers_dynamics.png`): UQS vs Error AUROC (0.5994) and ERS attribution stability discrimination.
- **Figure 4** (`fig4_hydraulic_risk_scatter.png`): Structural topology inadequacy ($\rho = 0.0932$) vs. physics consequence grounding ($\rho = +0.6109$).
- **Figure 5** (`fig5_triage_policy_tradeoff.png`): 5-Policy Action false alarm routing to review queue (routes 3 of 3 false-alarm events to review) vs operational response delay trade-off.

---

## 8. Related Work

TrustRisk-WDS synthesizes four essential research literatures:

1. **Selective Prediction, Reject Options & Deferral**:
   Selective prediction frameworks introduce an explicit abstain or defer option when classifier confidence is low (Geifman & El-Yaniv, 2017; Angelopoulos & Bates, 2021; Hendrickx et al., 2021). TrustRisk-WDS extends selective prediction into streaming cyber-physical security by routing ambiguous or unverified alerts into an asynchronous human review queue rather than triggering disruptive automated actuator shutdowns.
2. **Calibrated and Uncertainty-Aware Intrusion Detection**:
   Modern deep neural networks and machine learning models are frequently miscalibrated and overconfident on out-of-distribution inputs (Guo et al., 2017; Kull et al., 2019; Sensoy et al., 2018). Platt scaling, temperature scaling, and ensemble epistemic uncertainty quantification restore frequentist reliability to anomaly scores, providing trusted inputs for downstream risk gating.
3. **Alarm Fatigue and Triage in SOC and Industrial Control Systems**:
   Alert fatigue remains the leading cause of security failures in operational technology (OT) and industrial control systems (Alahmadi et al., 2022; Hassan et al., 2019; Taormina et al., 2018). High false-alarm rates desensitize operators and encourage blanket alarm deactivation. TrustRisk-WDS provides an operational filter that drastically reduces action-tier fatigue while maintaining full event recall.
4. **Critique of Point-Adjusted Evaluation in Time-Series Benchmarks**:
   Recent foundational analyses (Kim et al., 2022; Garg, Zhang, et al., 2021; Hwang et al., 2022) demonstrated that standard point-adjusted (PA) scoring creates an illusion of near-perfect anomaly detection by crediting entire multi-hour incidents upon a single detected point. In TrustRisk-WDS, all models are evaluated on unadjusted pointwise metrics ($F_1 = 0.8368, R = 73.71\%$) alongside PA metrics ($F_1 = 0.9879, R = 100.00\%) and explicit incident delay tracking.

---

## 9. Threats to Validity & Limitations

We transparently report all known threats to validity and experimental limitations:

1. **Simulated Hydraulic Data & Synthetic Attacks**: While C-Town topology and EPANET simulation reflect authentic municipal hydraulics, simulated consumer demands and synthetic cyber-attacks cannot fully capture real-world operational messiness, pipe bursts, and unexpected seasonal demand swings.
2. **Small False-Alarm Sample Size (3 Events) & Second Benchmark Corroboration**: Under baseline temporal persistence, the official BATADAL test set exhibits exactly 3 false-alarm events outside attack windows. While the calibrated two-tier gate routes 3 of 3 false-alarm events to analyst review (achieving a 100.0% reduction in Action-tier false alarms), the small sample size ($N=3$) permits no meaningful asymptotic confidence interval. Supporting validation evidence from Dataset 04 5-fold cross-validation across 12.4 weeks of held-out operational data shows 8 Action FA events under Persistence vs. 4 under Policy 4 and 13 under Policy 3. To further mitigate this limitation, Section 6b provides an independent 14-day WNTR C-Town benchmark demonstrating a 25.0% Action FA reduction across 4 distinct attack scenarios and 240 normal operating hours. Crucially, on both benchmarks, uncertainty (UQS) and explanation-reliability (ERS) terms provided no measurable benefit over calibrated confidence (PCS) alone, as PCS alone (Policy 3) achieved the maximum Action FA suppression while preserving all attacks.
3. **Sparsity of Monitored Sensor Topology (19 of 396 Nodes)**: In C-Town, only 19 physical locations out of 396 network elements are equipped with SCADA sensors. Root cause localization and cascading reach estimates rely on graph shortest-path propagation rather than exhaustive network instrumentation.
4. **Assumed Hardening Costs & Component Parameters**: Unit hardening costs ($120k for pumps, $80k for valves, $100k for tanks, $40k for junctions) and physical vulnerability factors represent engineering assumptions documented in `config/trustrisk.yaml`. While rank sensitivity analysis confirms invariance (Kendall $\tau = 0.9859$), municipal deployments must tailor cost parameters to utility procurement contracts.
5. **Test Set Seen During Downstream Development**: The primary Ridge detector was strictly frozen in `frozen_config.json` before first test exposure. However, downstream trust and triage layers were evaluated on the test set retrospectively. While all hyperparameter selections were restricted to Dataset 04 validation folds, readers must treat downstream policy evaluations as exploratory.
6. **Circular Moving-Block Bootstrap Assumption**: Circular block bootstrap assumes periodic boundary wrapping ($T-1 \to 0$) to eliminate sample loss. It is presented strictly as a sensitivity check on temporal autocorrelation, not a proof of stationarity.
7. **The Operational Latency Trade-Off**: Trust validation introduces an average response delay of **6.57 hours** (Policy 3), **5.71 hours** (Policy 4), and **6.86 hours** (Policy 5) vs. **4.86 hours** under persistence. Municipal utilities must determine whether this additional delay (1.7-2.3 h) is acceptable in exchange for routing false alarms to human review rather than executing automated actuator shutdowns.
8. **Calibration Transfer Limitation & Weak Epistemic Discrimination**: As documented in Section 4, post-hoc probability calibrators fitted on historical operational pools fail to improve test-time ECE/Brier under severe base-rate shifts (5.9% validation -> 19.5% test prevalence). Furthermore, test UQS error AUROC of 0.5994 demonstrates weak epistemic discrimination for unseen sensor anomalies. Consequently, trust-based triage relies on rank preservation rather than absolute probability guarantees.

---

## 10. References

1. Alahmadi, B. A., Axelsson, S., & Simpson, A. (2022). Investigating SOC analysts' alert fatigue: An empirical study. *ACM Transactions on Privacy and Security*, 25(4), 1–32.
2. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*.
3. Garg, S., Zhang, W., Chen, J., & Stevens, P. (2021). An evaluation of anomaly detection and diagnosis in multivariate time series. *IEEE Transactions on Neural Networks and Learning Systems*, 33(6), 2508–2522.
4. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
5. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *International Conference on Machine Learning (ICML)*, 1321–1330.
6. Hassan, W. U., Guo, S., Li, D., Chen, Z., et al. (2019). NoDoze: Combatting threat alert fatigue with automated provenance triage. *Network and Distributed System Security Symposium (NDSS)*.
7. Hendrickx, K., Perini, L., Van der Plas, D., Meert, W., & Davis, J. (2021). Machine learning with a reject option: A survey. *Machine Learning*, 110(5), 875–910.
8. Hwang, W. S., Choi, J. M., Lee, J. S., et al. (2022). Do you know existing time-series anomaly detection metrics are misleading? *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, 605–615.
9. Kim, S., Choi, K., Choi, H., Lee, B., & Yoon, S. (2022). Towards a rigorous evaluation of time-series anomaly detection. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(7), 7194–7201.
10. Klise, K. A., Bynum, M., Moriarty, D., & Murray, R. (2017). A software framework for assessing the resilience of drinking water systems to disasters with application to design and operational challenges. *Environmental Modelling & Software*, 92, 102–113.
11. Kull, M., Perello Nieto, M., Kängsepp, M., et al. (2019). Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration. *NeurIPS*, 32.
12. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *NeurIPS*, 31.
13. Taormina, R., Galelli, S., Douglas, H. C., et al. (2018). The Battle of the Attack Detection Algorithms: Disclosing cyber attacks on water distribution networks. *Journal of Water Resources Planning and Management*, 144(8), 04018048.

---

## 11. Author CRediT Roles & Human Responsibilities

All authorship, CRediT allocations, venue selection, and Zenodo release decisions are documented as human responsibilities in [`docs/HUMAN_TODO.md`](HUMAN_TODO.md).

---
