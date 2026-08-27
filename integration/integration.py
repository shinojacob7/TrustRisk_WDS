import pandas as pd
import numpy as np

class TrustRiskIntegrator:
    def __init__(self, risk_engine, trust_engine):
        """
        Combines the Cyber-Physical Risk Engine (Static Topology) 
        and the Trust Engine (Dynamic AI Metrics).
        """
        self.risk_engine = risk_engine
        self.trust_engine = trust_engine
        
        # Base risk report for all assets in the network
        self.static_risk_df = self.risk_engine.generate_risk_report()

    def compute_trust_score(self, conf, uncertainty, ers, w_conf=0.4, w_unc=0.3, w_ers=0.3):
        """
        Computes the unified Trust Score (TS).
        High confidence, low uncertainty, and high explanation reliability = High Trust.
        """
        # Normalize uncertainty (lower is better, so we invert it for the score)
        # Assuming uncertainty max is generally around 0.5 for probabilities
        normalized_unc = max(0, 1.0 - (uncertainty * 2)) 
        
        ts = (w_conf * conf) + (w_unc * normalized_unc) + (w_ers * ers)
        return min(max(ts, 0.0), 1.0) # Clamp between 0 and 1

    def compute_tdcri(self, trust_score, affected_node_id):
        """
        Computes the Trust-Aware Dynamic Cyber Risk Index (TDCRI).
        Combines static risk of the affected node with the trust of the AI alert.
        """
        # Get baseline risk for the specific node
        node_risk = self.static_risk_df[self.static_risk_df['Node_ID'] == affected_node_id]
        if node_risk.empty:
            baseline_risk = 0.5
        else:
            baseline_risk = node_risk['Baseline_Risk_Score'].values[0]
            
        # TDCRI heuristic: Risk magnitude amplified/modulated by alert trust.
        tdcri = baseline_risk * trust_score
        return tdcri

    def validate_alert(self, ts, tdcri, high_trust_threshold=0.7, high_risk_threshold=0.6):
        """
        Trust-Aware Alert Validation Rules
        """
        if ts >= high_trust_threshold and tdcri >= high_risk_threshold:
            return "Trigger Immediate Action (High Trust, High Risk)"
        elif ts >= high_trust_threshold and tdcri < high_risk_threshold:
            return "Monitor / Standard Logging (High Trust, Low Risk)"
        elif ts < high_trust_threshold and tdcri >= high_risk_threshold:
            return "Recommend Human Review (Low Trust, High Risk Potential!)"
        else:
            return "Discard / Monitor (Low Trust, Low Risk)"

    def optimize_security_investment(self):
        """
        Security Investment Optimization Engine
        Ranks assets by TrustRisk Protection Priority Index (TPPI).
        Here we use the baseline risk as a proxy for TPPI.
        """
        prioritized = self.static_risk_df.sort_values(by='Baseline_Risk_Score', ascending=False)
        return prioritized

if __name__ == "__main__":
    print("Integration module loaded successfully.")
