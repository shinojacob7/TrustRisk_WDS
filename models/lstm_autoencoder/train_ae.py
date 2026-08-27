import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size=12):
        self.window_size = window_size
        feature_cols = [c for c in df.columns if c not in ['DATETIME', 'ATT_FLAG']]
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df['ATT_FLAG'].values.astype(np.float32) if 'ATT_FLAG' in df.columns else np.zeros(len(df))
        
    def __len__(self):
        return len(self.features) - self.window_size
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y_label = self.labels[idx + self.window_size - 1] 
        return torch.tensor(x), torch.tensor(y_label)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.size()
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Repeat hidden state seq_len times to feed to decoder
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(hidden_repeated)
        return decoded

def train_autoencoder():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Loading temporal splits...")
    train_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'train.csv'))
    val_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'val.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'data', 'splits', 'test.csv'))
    
    window_size = 12
    batch_size = 64
    
    train_dataset = TimeSeriesDataset(train_df, window_size)
    val_dataset = TimeSeriesDataset(val_df, window_size)
    test_dataset = TimeSeriesDataset(test_df, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = len([c for c in train_df.columns if c not in ['DATETIME', 'ATT_FLAG']])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMAutoencoder(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
    print(f"Starting Unsupervised Training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                recon = model(x_batch)
                val_loss += criterion(recon, x_batch).item()
                
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss/len(val_loader):.4f}")
        
    print("\n--- Final Test Evaluation (Anomaly Scoring) ---")
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            recon = model(x_batch)
            # MSE per sample (mean over seq_len and features)
            mse_scores = torch.mean((recon - x_batch)**2, dim=[1, 2]).cpu().numpy()
            all_scores.extend(mse_scores)
            all_labels.extend(y_batch.numpy())
            
    # Calculate ROC-AUC if attacks are present in test set
    if sum(all_labels) > 0:
        auc = roc_auc_score(all_labels, all_scores)
        print(f"Test ROC-AUC: {auc:.4f}")
    else:
        print("No attack labels found in test set to compute AUC.")
    
    os.makedirs(os.path.join(base_dir, 'models', 'lstm_autoencoder'), exist_ok=True)
    save_path = os.path.join(base_dir, 'models', 'lstm_autoencoder', 'lstm_ae_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"LSTM Autoencoder saved to {save_path}")

if __name__ == "__main__":
    train_autoencoder()
