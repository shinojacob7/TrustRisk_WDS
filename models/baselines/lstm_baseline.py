import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class ChronologicalSequenceDataset(Dataset):
    def __init__(self, df, window_size=12, target_col='ATT_FLAG'):
        """
        Builds chronological sliding windows.
        Ensures no temporal leakage by predicting the label of the LAST time step in the window.
        """
        self.window_size = window_size
        
        # Exclude DATETIME and target from features
        feature_cols = [c for c in df.columns if c not in ['DATETIME', target_col]]
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[target_col].values.astype(np.float32)
        
    def __len__(self):
        return len(self.features) - self.window_size
        
    def __getitem__(self, idx):
        # Sequence from t to t + window_size
        x = self.features[idx : idx + self.window_size]
        # Label at t + window_size - 1 (the current timestamp we are monitoring)
        y = self.labels[idx + self.window_size - 1] 
        return torch.tensor(x), torch.tensor(y)

class TrustRiskBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(TrustRiskBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        
        # Classifier Head
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_out = lstm_out[:, -1, :] # Extract the last temporal step
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch).squeeze()
            loss = criterion(logits, y_batch)
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    val_loss /= len(loader)
    
    # Calculate Metrics
    # Handle cases where true labels might be all 0 (e.g., normal operating periods)
    if sum(all_labels) > 0:
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        f1, precision, recall, roc_auc = 0.0, 0.0, 0.0, 0.0
        
    return val_loss, f1, precision, recall, roc_auc

def train_baseline():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Loading temporal splits...")
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'train.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'val.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'test.csv'))
    
    window_size = 12 # e.g., 12 hours of historical context
    batch_size = 64
    
    train_dataset = ChronologicalSequenceDataset(train_df, window_size)
    val_dataset = ChronologicalSequenceDataset(val_df, window_size)
    test_dataset = ChronologicalSequenceDataset(test_df, window_size)
    
    # It is acceptable to shuffle windows during training to break batch correlation,
    # but the sequence inside the window is strictly temporal.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len([c for c in train_df.columns if c not in ['DATETIME', 'ATT_FLAG']])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TrustRiskBiLSTM(input_dim=input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    print(f"Starting Training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        val_loss, f1, prec, rec, auc = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {f1:.4f} | Val AUC: {auc:.4f}")
        
    print("\n--- Final Test Evaluation ---")
    test_loss, t_f1, t_prec, t_rec, t_auc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test F1: {t_f1:.4f} | Precision: {t_prec:.4f} | Recall: {t_rec:.4f} | ROC-AUC: {t_auc:.4f}")
    
    save_path = os.path.join(base_dir, 'models', 'baselines', 'bilstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Strict temporal baseline model saved to {save_path}")

if __name__ == "__main__":
    train_baseline()
