import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix
)
import json
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from models.lstm_autoencoder.train_ae import LSTMAutoencoder, TimeSeriesDataset

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

def generate_metrics():
    print("Loading Validation & Test data, and Unsupervised LSTM Autoencoder...")
    
    # Use preprocessed splits consistently if possible, else fallback to splits/
    val_path = os.path.join(base_dir, 'data', 'processed', 'val_processed.csv')
    test_path = os.path.join(base_dir, 'data', 'processed', 'test_processed.csv')
    
    if not os.path.exists(val_path):
        val_path = os.path.join(base_dir, 'data', 'splits', 'val.csv')
        test_path = os.path.join(base_dir, 'data', 'splits', 'test.csv')
        
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    input_dim = len([c for c in test_df.columns if c not in ['DATETIME', 'ATT_FLAG']])
    model = LSTMAutoencoder(input_dim=input_dim)
    
    model_path = os.path.join(base_dir, 'models', 'lstm_autoencoder', 'lstm_ae_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    val_dataset = TimeSeriesDataset(val_df, window_size=12)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    test_dataset = TimeSeriesDataset(test_df, window_size=12)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def get_scores(loader):
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                recon = model(x_batch)
                mse_scores = torch.mean((recon - x_batch)**2, dim=[1, 2]).numpy()
                all_scores.extend(mse_scores)
                all_labels.extend(y_batch.numpy())
        return np.array(all_scores), np.array(all_labels)

    # Stage A Fix: Threshold chosen exclusively on Validation Data
    val_scores, val_labels = get_scores(val_loader)
    
    # We sweep threshold on validation data to maximize Point-Adjusted F1
    best_f1 = 0
    best_thresh = 0
    # Search over quantiles of the MSE scores to find a good threshold
    thresholds = np.percentile(val_scores, np.linspace(80, 99.9, 200))
    for t in thresholds:
        preds = (val_scores >= t).astype(int)
        pa_preds = point_adjust(val_labels, preds)
        f1 = f1_score(val_labels, pa_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Selected Validation Threshold (PA F1): {best_thresh:.4f} (Val F1: {best_f1:.4f})")
    
    # Now evaluate on Test Data using the frozen threshold
    test_scores, test_labels = get_scores(test_loader)
    
    y_pred_raw = (test_scores >= best_thresh).astype(int)
    y_pred_pa = point_adjust(test_labels, y_pred_raw)
    
    acc = accuracy_score(test_labels, y_pred_pa)
    prec = precision_score(test_labels, y_pred_pa, zero_division=0)
    rec = recall_score(test_labels, y_pred_pa)
    f1 = f1_score(test_labels, y_pred_pa)
    auc = roc_auc_score(test_labels, test_scores)
    pr_auc = average_precision_score(test_labels, test_scores)
    cm = confusion_matrix(test_labels, y_pred_pa)
    
    edr, mean_delay, det_ev, tot_ev, fa = get_event_metrics(test_labels, y_pred_raw)
    
    print("\n--- UNSUPERVISED LSTM-AE FINAL EVALUATION METRICS ---")
    print(f"Accuracy (PA):  {acc:.4f}")
    print(f"Precision (PA): {prec:.4f}")
    print(f"Recall (PA):    {rec:.4f}")
    print(f"F1 Score (PA):  {f1:.4f}")
    print(f"ROC-AUC:        {auc:.4f}")
    print(f"PR-AUC:         {pr_auc:.4f}")
    print(f"Event Detection Rate: {edr:.2f} ({det_ev}/{tot_ev})")
    print(f"Mean Det. Delay:      {mean_delay:.2f} frames")
    print(f"False Alarms (Event): {fa}")
    
    metrics_dict = {
        "Accuracy_PA": round(acc, 4),
        "Precision_PA": round(prec, 4),
        "Recall_PA": round(rec, 4),
        "F1_Score_PA": round(f1, 4),
        "ROC_AUC": round(auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Event_Detection_Rate": round(edr, 4),
        "Detected_Events": det_ev,
        "Total_Events": tot_ev,
        "Mean_Detection_Delay": round(mean_delay, 4),
        "False_Alarm_Events": fa,
        "Confusion_Matrix": cm.tolist(),
        "Threshold": round(best_thresh, 4)
    }
    
    os.makedirs(os.path.join(base_dir, 'docs'), exist_ok=True)
    with open(os.path.join(base_dir, 'docs', 'ae_evaluation_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    print("Metrics saved to docs/ae_evaluation_metrics.json")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    generate_metrics()
