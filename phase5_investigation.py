import pandas as pd
import numpy as np
import json
import os

def main():
    # Load old corrupted Pipeline Data, but strictly extract the virgin GNN probability
    df = pd.read_csv('experiments/final/trust_pipeline.csv')
    labels = df['true_label'].values
    probs = df['p_attack_gnn'].values
    
    # Load Frozen Threshold
    with open('experiments/final/final_metrics.json', 'r') as f:
        metrics = json.load(f)
    thresh = metrics['Threshold']
    
    # PHASE 5: INVESTIGATE FALSE NEGATIVES (Attack-event diagnostic)
    print("PHASE 5: ATTACK EVENT TIMELINE DIAGNOSTIC (using p_attack_gnn)")
    print(f"Frozen Threshold: {thresh}")
    attack_indices = np.where(labels == 1)[0]
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    
    for i, event in enumerate(events):
        ev_probs = probs[event]
        ev_max = np.max(ev_probs)
        ev_mean = np.mean(ev_probs)
        detected = np.any(ev_probs >= thresh)
        duration = len(event)
        
        delay_str = "Missed completely"
        if detected:
            delay = np.argmax(ev_probs >= thresh)
            delay_str = f"Detected at frame {delay}"
            
        print(f"\nEvent {i+1} [Frames {event[0]}-{event[-1]} | Duration: {duration} hr]:")
        print(f"   Max Prob: {ev_max:.4f} | Mean Prob: {ev_mean:.4f}")
        print(f"   Status: {'DETECTED' if detected else 'MISSED'} | Delay: {delay_str}")

if __name__ == "__main__":
    main()
