import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_wds_datasets(input_dir='data/splits', output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- [Phase 1] Loading raw benchmark datasets ---")
    train_df = pd.read_csv(os.path.join(input_dir, 'train.csv'))
    val_df   = pd.read_csv(os.path.join(input_dir, 'val.csv'))
    test_df  = pd.read_csv(os.path.join(input_dir, 'test.csv'))

    target_col = 'ATT_FLAG'
    datetime_col = 'DATETIME'
    
    # Identify binary actuator columns (pumps/valves) and continuous sensor columns
    binary_cols = [c for c in train_df.columns if c.startswith('S_')]
    non_feature_cols = [target_col, datetime_col]
    continuous_cols = [
        c for c in train_df.columns 
        if c not in binary_cols and c not in non_feature_cols
    ]

    print(f"Total features: {len(binary_cols) + len(continuous_cols)}")
    print(f"Continuous features ({len(continuous_cols)}): {continuous_cols[:5]}...")
    print(f"Binary features ({len(binary_cols)}): {binary_cols}")

    # 1. Fit scaler ONLY on training data continuous features (prevents data leakage)
    print("\nFitting StandardScaler strictly on training continuous variables...")
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])

    # 2. Transform all splits using the training baseline
    train_proc = train_df.copy()
    val_proc   = val_df.copy()
    test_proc  = test_df.copy()

    train_proc[continuous_cols] = scaler.transform(train_df[continuous_cols])
    val_proc[continuous_cols]   = scaler.transform(val_df[continuous_cols])
    test_proc[continuous_cols]  = scaler.transform(test_df[continuous_cols])

    # 3. Enforce binary integrity (strictly 0 or 1)
    for c in binary_cols:
        train_proc[c] = np.clip(np.round(train_proc[c].values), 0, 1).astype(int)
        val_proc[c]   = np.clip(np.round(val_proc[c].values), 0, 1).astype(int)
        test_proc[c]  = np.clip(np.round(test_proc[c].values), 0, 1).astype(int)

    # 4. Save processed datasets
    train_path = os.path.join(output_dir, 'train_processed.csv')
    val_path   = os.path.join(output_dir, 'val_processed.csv')
    test_path  = os.path.join(output_dir, 'test_processed.csv')

    train_proc.to_csv(train_path, index=False)
    val_proc.to_csv(val_path, index=False)
    test_proc.to_csv(test_path, index=False)

    print(f"Saved processed training data -> {train_path} ({len(train_proc)} rows)")
    print(f"Saved processed validation data -> {val_path} ({len(val_proc)} rows)")
    print(f"Saved processed testing data    -> {test_path} ({len(test_proc)} rows)")
    print("Preprocessing completed successfully.\n")

if __name__ == '__main__':
    prepare_wds_datasets()
