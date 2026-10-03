import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score, confusion_matrix)

def point_adjust(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    adjusted_preds = preds.copy()
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0: return adjusted_preds
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    for event in events:
        if np.any(preds[event] == 1):
            adjusted_preds[event] = 1
    return adjusted_preds

def get_event_metrics(labels, preds):
    labels = np.array(labels)
    preds = np.array(preds)
    
    attack_indices = np.where(labels == 1)[0]
    if len(attack_indices) == 0:
        fp_mask = (preds == 1)
        fp_indices = np.where(fp_mask)[0]
        false_alarms = 0
        if len(fp_indices) > 0:
            fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
            false_alarms = len(fp_events)
        return 0.0, 0.0, 0, 0, false_alarms
        
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    
    detected_events = 0
    total_delay = 0
    
    for event in events:
        event_preds = preds[event]
        if np.any(event_preds == 1):
            detected_events += 1
            delay = np.argmax(event_preds == 1)
            total_delay += delay
            
    fp_mask = (preds == 1) & (labels == 0)
    fp_indices = np.where(fp_mask)[0]
    false_alarms = 0
    if len(fp_indices) > 0:
        fp_events = np.split(fp_indices, np.where(np.diff(fp_indices) != 1)[0] + 1)
        false_alarms = len(fp_events)
        
    edr = detected_events / len(events)
    mean_delay = total_delay / detected_events if detected_events > 0 else 0
    return edr, mean_delay, detected_events, len(events), false_alarms

def main():
    # Load Virgin Pipeline Data
    df = pd.read_csv('experiments/final/trust_pipeline_virgin.csv')
    labels = df['true_label'].values
    probs = df['p_attack'].values
    
    # Load Frozen Threshold
    with open('experiments/final/final_metrics.json', 'r') as f:
        metrics = json.load(f)
    thresh = metrics['Threshold']
    
    # Raw predictions
    preds = (probs >= thresh).astype(int)
    
    # POINTWISE METRICS
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    p_prec = precision_score(labels, preds, zero_division=0)
    p_rec = recall_score(labels, preds, zero_division=0)
    p_f1 = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)
    roc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    
    # POINT-ADJUSTED (PA) METRICS
    pa_preds = point_adjust(labels, preds)
    pa_tn, pa_fp, pa_fn, pa_tp = confusion_matrix(labels, pa_preds).ravel()
    pa_prec = precision_score(labels, pa_preds, zero_division=0)
    pa_rec = recall_score(labels, pa_preds, zero_division=0)
    pa_f1 = f1_score(labels, pa_preds, zero_division=0)
    
    edr, mean_delay, det_ev, tot_ev, fa = get_event_metrics(labels, preds)
    
    print("="*50)
    print("A. POINTWISE METRICS (RAW)")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Precision: {p_prec:.4f} | Recall: {p_rec:.4f} | F1: {p_f1:.4f}")
    print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")
    print("\nB. EVENT/POINT-ADJUSTED (PA) METRICS")
    print(f"PA TP: {pa_tp} | PA FP: {pa_fp} | PA FN: {pa_fn} | PA TN: {pa_tn}")
    print(f"PA Precision: {pa_prec:.4f} | PA Recall: {pa_rec:.4f} | PA F1: {pa_f1:.4f}")
    print(f"Event Detection Rate: {edr:.4f} ({det_ev}/{tot_ev})")
    print(f"False Alarm Events: {fa}")
    print(f"Mean Det Delay: {mean_delay:.2f} frames")
    print("="*50)
    
    # Save to virgin JSON
    out = {
        "Threshold": thresh,
        "Pointwise_CM": [int(tn), int(fp), int(fn), int(tp)],
        "Pointwise_Precision": round(p_prec, 4),
        "Pointwise_Recall": round(p_rec, 4),
        "Pointwise_F1": round(p_f1, 4),
        "ROC_AUC": round(roc, 4),
        "PR_AUC": round(pr_auc, 4),
        "PA_CM": [int(pa_tn), int(pa_fp), int(pa_fn), int(pa_tp)],
        "PA_Precision": round(pa_prec, 4),
        "PA_Recall": round(pa_rec, 4),
        "PA_F1": round(pa_f1, 4),
        "Event_Detection_Rate": round(edr, 4),
        "Detected_Events": int(det_ev),
        "Total_Events": int(tot_ev),
        "False_Alarm_Events": int(fa),
        "Mean_Detection_Delay": round(mean_delay, 4)
    }
    with open('experiments/final/virgin_baseline_metrics.json', 'w') as f:
        json.dump(out, f, indent=4)
        
    # PHASE 5: INVESTIGATE FALSE NEGATIVES (Attack-event diagnostic)
    print("\nPHASE 5: ATTACK EVENT TIMELINE DIAGNOSTIC")
    attack_indices = np.where(labels == 1)[0]
    events = np.split(attack_indices, np.where(np.diff(attack_indices) != 1)[0] + 1)
    
    for i, event in enumerate(events):
        ev_probs = probs[event]
        ev_max = np.max(ev_probs)
        ev_mean = np.mean(ev_probs)
        detected = np.any(ev_probs >= thresh)
        duration = len(event)
        
        delay_str = "Missed"
        if detected:
            delay = np.argmax(ev_probs >= thresh)
            delay_str = f"{delay} frames"
            
        print(f"Event {i+1} [Frames {event[0]}-{event[-1]} | {duration} hr]:")
        print(f"   Max Prob: {ev_max:.4f} | Mean Prob: {ev_mean:.4f}")
        print(f"   Status: {'DETECTED' if detected else 'MISSED'} | Delay: {delay_str}")

if __name__ == "__main__":
    main()
