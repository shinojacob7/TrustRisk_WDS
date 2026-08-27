import numpy as np

class TrustRiskIntegrator:
    """
    Phase 5: TrustRisk Intelligence Integration Layer
    Formally integrates the AI Trust metrics with Cyber-Physical Risk metrics to 
    produce the Trust-Aware Dynamic Cyber Risk Index (TDCRI) and solve the 
    Security Investment Optimization Problem (TPPI).
    """
    def __init__(self, w_conf=0.4, w_unc=0.3, w_ers=0.3):
        # Weights for the Trust Score components
        assert np.isclose(w_conf + w_unc + w_ers, 1.0), "Weights must sum to 1.0"
        self.w_conf = w_conf
        self.w_unc = w_unc
        self.w_ers = w_ers

    def compute_trust_score(self, attack_prob, predictive_entropy, max_entropy, ers):
        """
        Computes the formal Trust Score (TS).
        - Confidence: Distance of prob from decision boundary (0.5), mapped to [0,1].
        - Uncertainty Penalty: Normalized predictive entropy mapped to [0,1].
        - ERS: Explanation Reliability Score [0,1].
        """
        # 1. Confidence Score (How decisive is the probability?)
        confidence = abs(attack_prob - 0.5) * 2.0  # 0.5 -> 0.0, 1.0 -> 1.0, 0.0 -> 1.0
        
        # 2. Uncertainty Score (How little entropy/variance is in the MC Dropout?)
        # Low entropy = High Certainty (Score near 1.0)
        certainty = 1.0 - (predictive_entropy / (max_entropy + 1e-9))
        certainty = np.clip(certainty, 0.0, 1.0)
        
        # 3. Trust Score (Weighted Combination)
        ts = (self.w_conf * confidence) + (self.w_unc * certainty) + (self.w_ers * ers)
        return np.clip(ts, 0.0, 1.0)

    def compute_tdcri(self, attack_prob, trust_score, mdci, ass, iss, cri):
        """
        Computes the Trust-Aware Dynamic Cyber Risk Index (TDCRI).
        Combines the Dynamic Threat (Prob * Trust) with the Static Cyber-Physical Risk (Chain of Vulnerability).
        """
        # Baseline Physical Risk (Chain formulation: Asset Criticality * Susceptibility * Impact * Cascading Risk)
        # Assuming all inputs are normalized [0, 1] from the Risk Engine
        physical_risk = mdci * ass * iss * cri
        
        # Dynamic Risk is the Trusted Probability of Attack
        dynamic_threat = attack_prob * trust_score
        
        # TDCRI fuses both dimensions
        tdcri = dynamic_threat * physical_risk
        return tdcri

    def optimize_security_investments(self, assets, budget):
        """
        Solves the Trust-aware Prioritized Protection Index (TPPI) Optimization Problem.
        Rather than simple sorting, this solves the 0/1 Knapsack Problem for Security Allocation.
        
        assets: list of dicts [{'node_id': 'T1', 'tdcri': 0.85, 'protection_cost': 100}, ...]
        budget: total available security budget (e.g., 250)
        
        Returns: Selected assets that maximize total risk reduction (TDCRI) under budget constraints.
        """
        n = len(assets)
        if n == 0:
            return []
            
        # Extract values (V) and weights/costs (W)
        V = [asset['tdcri'] for asset in assets]
        W = [int(asset['protection_cost']) for asset in assets]
        B = int(budget)
        
        # DP Table for 0/1 Knapsack
        dp = np.zeros((n + 1, B + 1))
        
        for i in range(1, n + 1):
            for w in range(B + 1):
                if W[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - W[i-1]] + V[i-1])
                else:
                    dp[i][w] = dp[i-1][w]
                    
        # Backtrack to find selected assets
        selected_assets = []
        w = B
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_assets.append(assets[i-1])
                w -= W[i-1]
                
        # Return sorted by TDCRI (highest first) for reporting
        selected_assets.sort(key=lambda x: x['tdcri'], reverse=True)
        return selected_assets

if __name__ == "__main__":
    # Quick test of the Optimization Engine
    integrator = TrustRiskIntegrator()
    
    sample_assets = [
        {'node_id': 'T1', 'tdcri': 0.92, 'protection_cost': 120},
        {'node_id': 'PU1', 'tdcri': 0.75, 'protection_cost': 80},
        {'node_id': 'V2', 'tdcri': 0.40, 'protection_cost': 50},
        {'node_id': 'J280', 'tdcri': 0.65, 'protection_cost': 70}
    ]
    budget = 200
    
    selected = integrator.optimize_security_investments(sample_assets, budget)
    print("TPPI Optimization Test (Budget: 200):")
    for asset in selected:
        print(f"Deploy defense to {asset['node_id']} | Risk Mitigated: {asset['tdcri']} | Cost: {asset['protection_cost']}")
