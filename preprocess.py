"""
IIOT NETWORK DATA PREPROCESSING MODULE
======================================
Purpose:
    Cleans, deduplicates, and splits the raw WUSTL-IIoT-2021 tabular network 
    dataset into robust training and validation matrices.
    
Key Interventions:
    - Drops architectural identity shortcuts ('SrcAddr', 'DstAddr', 'StartTime', 'LastTime').
    - Drops service application port shortcuts ('Sport', 'Dport') to enforce behavioral learning.
    - Suppresses duplicate records to eliminate sample memorization.
    - Fits the StandardScaler strictly on X_train to prevent global data leakage.
    - Compiles structural edge indexes and masked nodes specifically for the GNN encoder.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    df = df.drop_duplicates().reset_index(drop=True)
    
    target_col = 'Traffic' if 'Traffic' in df.columns else 'Label'
    
    src_ips = df['SrcAddr'].astype(str).values if 'SrcAddr' in df.columns else np.arange(len(df))
    dst_ips = df['DstAddr'].astype(str).values if 'DstAddr' in df.columns else np.arange(len(df))
    
    shortcut_features = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId', 'Sport', 'Dport']
    df_clean = df.drop(columns=[col for col in shortcut_features if col in df.columns], errors='ignore')
    
    y = df_clean[target_col].values
    X = df_clean.drop(columns=[target_col], errors='ignore')
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        X.values, y_encoded, np.arange(len(y)), test_size=0.3, stratify=y_encoded, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.nan_to_num(X_train_raw))
    X_test = scaler.transform(np.nan_to_num(X_test_raw))
    
    unique_nodes = np.unique(np.concatenate([src_ips, dst_ips]))
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    
    edge_index = np.array([
        [node_map[src] for src in src_ips],
        [node_map[dst] for dst in dst_ips]
    ], dtype=np.int64)
    
    all_features = np.zeros((len(df), X_train.shape[1]))
    all_features[train_idx] = X_train
    all_features[test_idx] = X_test
    
    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    graph_data = {
        'num_nodes': len(unique_nodes),
        'edge_index': edge_index,
        'edge_features': all_features,
        'train_mask': train_mask,
        'test_mask': test_mask
    }
    
    return X_train, X_test, y_train, y_test, graph_data, label_encoder