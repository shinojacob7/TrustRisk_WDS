import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def build_datasets():
    print("Loading raw BATADAL datasets...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths
    path_03 = os.path.join(base_dir, 'data', 'raw', 'BATADAL_dataset03.csv')
    path_04 = os.path.join(base_dir, 'data', 'raw', 'BATADAL_dataset04.csv')
    path_test = os.path.join(base_dir, 'data', 'raw', 'BATADAL_test_dataset_labeled.csv')
    
    df_03 = pd.read_csv(path_03, skipinitialspace=True)
    df_04 = pd.read_csv(path_04, skipinitialspace=True)
    df_test = pd.read_csv(path_test, skipinitialspace=True)
    
    # Standardize column names
    df_03.columns = df_03.columns.str.strip()
    df_04.columns = df_04.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    # Ensure ATT_FLAG exists and clean up labels (-999 -> 0)
    if 'ATT_FLAG' not in df_03.columns:
        df_03['ATT_FLAG'] = 0
    else:
        df_03['ATT_FLAG'] = df_03['ATT_FLAG'].replace(-999, 0)
        
    if 'ATT_FLAG' not in df_04.columns:
        df_04['ATT_FLAG'] = 0
    else:
        df_04['ATT_FLAG'] = df_04['ATT_FLAG'].replace(-999, 0)
        
    df_test['ATT_FLAG'] = df_test['ATT_FLAG'].replace(-999, 0)

    # Parse datetimes (dateutil fallback is slow, format specifies exact string matching if possible, but letting pandas infer is safer here since batadal uses dd/mm/yy HH:MM)
    df_03['DATETIME'] = pd.to_datetime(df_03['DATETIME'], dayfirst=True)
    df_04['DATETIME'] = pd.to_datetime(df_04['DATETIME'], dayfirst=True)
    df_test['DATETIME'] = pd.to_datetime(df_test['DATETIME'], dayfirst=True)
    
    df_03 = df_03.sort_values('DATETIME').reset_index(drop=True)
    df_04 = df_04.sort_values('DATETIME').reset_index(drop=True)
    df_test = df_test.sort_values('DATETIME').reset_index(drop=True)
    
    # Feature Classification
    target = 'ATT_FLAG'
    timestamp = 'DATETIME'
    
    all_features = [c for c in df_03.columns if c not in [target, timestamp]]
    binary_features = [c for c in all_features if c.startswith('S_')]
    continuous_features = [c for c in all_features if c not in binary_features]
    
    print(f"Identified {len(continuous_features)} continuous features and {len(binary_features)} binary states.")
    
    # Define train/validation/test protocol
    # To train supervised models (like GNN-GRU), the model needs to see attacks.
    # Dataset 03 has 0 attacks. Dataset 04 has attacks.
    # We will combine 03 and the FIRST HALF of 04 into TRAIN.
    # We will use the SECOND HALF of 04 as VALIDATION.
    # This prevents temporal leakage while ensuring both splits see normal and attack data.
    
    split_04_idx = int(len(df_04) * 0.5)
    df_04_train = df_04.iloc[:split_04_idx].copy()
    df_04_val = df_04.iloc[split_04_idx:].copy()
    
    df_train = pd.concat([df_03, df_04_train], ignore_index=True).sort_values('DATETIME').reset_index(drop=True)
    df_val = df_04_val.copy()
    
    print(f"Train size: {len(df_train)} (includes {int(df_train['ATT_FLAG'].sum())} attacks)")
    print(f"Val size: {len(df_val)} (includes {int(df_val['ATT_FLAG'].sum())} attacks)")
    print(f"Test size: {len(df_test)} (includes {int(df_test['ATT_FLAG'].sum())} attacks)")
    
    # Scale continuous features
    scaler = StandardScaler()
    
    # Fit ONLY on training data
    scaler.fit(df_train[continuous_features])
    
    df_train[continuous_features] = scaler.transform(df_train[continuous_features])
    df_val[continuous_features] = scaler.transform(df_val[continuous_features])
    df_test[continuous_features] = scaler.transform(df_test[continuous_features])
    
    # Ensure binary states are int/float 0.0 or 1.0 (no scaling)
    for bf in binary_features:
        df_train[bf] = df_train[bf].astype(float)
        df_val[bf] = df_val[bf].astype(float)
        df_test[bf] = df_test[bf].astype(float)

    # Save to data/splits/
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    df_train.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    print("Preprocessing complete. Strict chronological splits with attack exposure saved to data/splits/")

if __name__ == "__main__":
    build_datasets()
