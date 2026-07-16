"""
ZERO-DAY TOPOLOGICAL PREPROCESSING FRAMEWORK (BINARY SCHEME)
===========================================================
Description:
    Processes the WUSTL-IIOT-2021 telemetry by stripping flat tabular shortcut leaks.
    Forces structural context tracking by engineering topological degree shapes.
    Standardizes label output to binary (0 = Normal, 1 = Anomaly/Attack).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    target_col = 'Traffic' if 'Traffic' in df.columns else 'Label'
    leak_cols = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    src_ips = df['SrcAddr'].astype(str).values if 'SrcAddr' in df.columns else np.arange(len(df))
    dst_ips = df['DstAddr'].astype(str).values if 'DstAddr' in df.columns else np.arange(len(df))
    
    # Topological structural feature engineering
    unique_nodes, node_counts = np.unique(np.concatenate([src_ips, dst_ips]), return_counts=True)
    node_degree_map = dict(zip(unique_nodes, node_counts))
    
    df['src_degree'] = [node_degree_map[src] for src in src_ips]
    df['dst_degree'] = [node_degree_map[dst] for dst in dst_ips]
    
    active_leaks = [c for c in leak_cols if c in df.columns]
    df_clean = df.drop(columns=active_leaks, errors='ignore')
    
    # FIX: Clean string arrays using pandas vectorized methods before converting to numpy
    # This prevents the "TypeError: string operation on non-string array" 
    y_series = df_clean[target_col].astype(str).str.strip().str.lower()
    binary_labels = np.where(y_series == 'normal', 0, 1)
    
    X = df_clean.drop(columns=[target_col], errors='ignore')
    
    # Handle remaining non-numeric columns
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    feature_cols = list(X.columns)
    
    indices = np.arange(len(df))
    # Stratified Split on binary labels ensuring class representation balance
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.5, 
        stratify=binary_labels, 
        random_state=42
    )
    
    X_train, y_train = X_scaled[train_idx], binary_labels[train_idx]
    X_test, y_test = X_scaled[test_idx], binary_labels[test_idx]
    
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    edge_index = np.array([
        [node_map[src] for src in src_ips],
        [node_map[dst] for dst in dst_ips]
    ], dtype=np.int64)
    
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    graph_data = {
        'edge_index': edge_index,
        'edge_features': X_scaled,
        'num_nodes': len(unique_nodes),
        'train_mask': train_mask,
        'test_mask': test_mask
    }
    
    # FIX: Return all 7 values (including scaler and feature_cols) expected by main.py
    return X_train, X_test, y_train, y_test, graph_data, scaler, feature_cols