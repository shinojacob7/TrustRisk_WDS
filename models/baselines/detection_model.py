import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class BATADALSequenceDataset(Dataset):
    def __init__(self, df, window_size=10, target_col='ATT_FLAG'):
        self.window_size = window_size
        self.features = df.drop(columns=[target_col]).values.astype(np.float32)
        self.labels = df[target_col].values.astype(np.float32)
        
    def __len__(self):
        return len(self.features) - self.window_size
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        # Label for the sequence is the label of the last time step in the window
        y = self.labels[idx + self.window_size - 1] 
        return torch.tensor(x), torch.tensor(y)

class TrustRiskBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(TrustRiskBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiLSTM Layer
        # Note: We enable dropout here. For MC Dropout (Phase 4), we will leave dropout active during inference!
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
        # x shape: (batch, window_size, features)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]
        
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits

def train_model(data_path, epochs=5, batch_size=64, window_size=10):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Simple split (80% train, 20% val)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_dataset = BATADALSequenceDataset(train_df, window_size)
    val_dataset = BATADALSequenceDataset(val_df, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 43 features in BATADAL
    input_dim = train_df.shape[1] - 1 
    model = TrustRiskBiLSTM(input_dim=input_dim)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                logits = model(x_batch).squeeze()
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
                
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val Acc: {correct/total:.4f}")
        
    os.makedirs('src/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'src/saved_models/bilstm_model.pth')
    print("Model saved to src/saved_models/bilstm_model.pth")
    return model

if __name__ == "__main__":
    train_model('data/processed/BATADAL_balanced_scaled.csv', epochs=3)
