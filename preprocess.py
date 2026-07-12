"""
ZERO-DAY TOPOLOGICAL PREPROCESSING FRAMEWORK
===========================================
Description:
    Processes the WUSTL-IIOT-2021 telemetry by stripping flat tabular shortcut leaks.
    Forces structural context tracking by engineering topological degree shapes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    target_col = 'Traffic' if 'Traffic' in df.columns else 'Label'
    leak_cols = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    
    src_ips = df['SrcAddr'].astype(str).values if 'SrcAddr' in df.columns else np.arange(len(df))
    dst_ips = df['DstAddr'].astype(str).values if 'DstAddr' in df.columns else np.arange(len(df))
    
    # Structural Engineering: Track topological degree distribution directly
    unique_nodes, node_counts = np.unique(np.concatenate([src_ips, dst_ips]), return_counts=True)
    node_degree_map = dict(zip(unique_nodes, node_counts))
    
    df['src_degree'] = [node_degree_map[src] for src in src_ips]
    df['dst_degree'] = [node_degree_map[dst] for dst in dst_ips]
    
    active_leaks = [c for c in leak_cols if c in df.columns]
    df_clean = df.drop(columns=active_leaks, errors='ignore')
    
    y = df_clean[target_col].astype(str).str.strip().values
    X = df_clean.drop(columns=[target_col], errors='ignore')
    
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(0))
    
    normal_val = 'Normal' if 'Normal' in label_encoder.classes_ else label_encoder.classes_[0]
    normal_idx = label_encoder.transform([normal_val])[0]
    binary_labels = np.where(y_encoded == normal_idx, 0, 1)
    
    indices = np.arange(len(df))
    # Stratified Split ensuring identical class baselines across validation layers
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.5, 
        stratify=binary_labels, 
        random_state=42
    )
    
    X_train, y_train = X_scaled[train_idx], y_encoded[train_idx]
    X_test, y_test = X_scaled[test_idx], y_encoded[test_idx]
    
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
    
    return X_train, X_test, y_train, y_test, graph_data, label_encoder