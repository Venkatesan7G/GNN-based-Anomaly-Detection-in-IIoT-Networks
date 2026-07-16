"""
INTEGRATED PRODUCTION TRAINING & SERIALIZATION ENGINE (STABILIZED BINARY PIPELINE)
=================================================================================
Description:
    Orchestrates the training loops for the classic classifiers, deep learning baselines,
    and self-supervised Anomal-E model on binary classifications.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import TensorDataset, DataLoader

from preprocess import load_and_preprocess_data
from models import (
    get_tabular_models, 
    CNNModel, 
    LSTMModel, 
    EGraphSAGEEncoder, 
    DeepGraphInfomaxAnomalE, 
    get_anomal_e_detector,
    FullAnomalEPipeline
)
from evaluate import evaluate_and_save

def main():
    csv_path = "data/wustl_iiot_2021.csv"
    if not os.path.exists(csv_path):
        print(f"[Error] Source dataset not found at {csv_path}. Run run_pipeline.py first.")
        return
        
    print("[Step 1/4] Preprocessing WUSTL-IIoT-2021 data into binary schemas...")
    X_train, X_test, y_train, y_test, graph_data, scaler, feature_cols = load_and_preprocess_data(csv_path)
    
    # Save objects needed for aligned generalization in cross_validate.py
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # -----------------------------------------------------------------
    # Step 2: Tabular Base Testing Loop & Serialization
    # -----------------------------------------------------------------
    print("\n[Step 2/4] Training Classic Tabular Architectural Baselines...")
    tabular_models = get_tabular_models()
    for name, model in tabular_models.items():
        print(f"  --> Processing execution path: {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        evaluate_and_save(name, y_test, preds)
        
        sanitized_filename = name.lower().replace(" ", "_")
        joblib.dump(model, f"models/{sanitized_filename}_model.pkl")
        
    # -----------------------------------------------------------------
    # Step 3: Deep Learning Sequence Loops & Serialization
    # -----------------------------------------------------------------
    print("\n[Step 3/4] Running Specialized Deep Learning Sequence Loops...")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), 
        batch_size=256, shuffle=True
    )
    
    dl_models = {
        "CNN": CNNModel(X_train.shape[1], 2).to(device),
        "LSTM": LSTMModel(X_train.shape[1], 2).to(device)
    }
    
    for name, model in dl_models.items():
        print(f"  --> Activating weight optimizations for instance: {name}")
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(5):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        test_preds_list = []
        eval_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)), 
            batch_size=4096, shuffle=False
        )
        
        with torch.no_grad():
            for batch in eval_loader:
                batch_x = batch[0].to(device)
                outputs = model(batch_x)
                batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                test_preds_list.append(batch_preds)
                
        preds = np.concatenate(test_preds_list)
        evaluate_and_save(name, y_test, preds)
        
        del test_preds_list
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), f"models/{name.lower()}_model.pt")

    # -----------------------------------------------------------------
    # Step 4: Self-Supervised Anomal-E Loop & Serialization
    # -----------------------------------------------------------------
    print("\n[Step 4/4] Activating Anomal-E Unsupervised Graph Processing Loop...")
    edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long).to(device)
    edge_feats = torch.tensor(graph_data['edge_features'], dtype=torch.float32).to(device)
    train_mask = graph_data['train_mask']
    test_mask = graph_data['test_mask']
    
    embedding_dim = 32
    encoder_net = EGraphSAGEEncoder(graph_data['num_nodes'], X_train.shape[1], embedding_dim).to(device)
    dgi_model = DeepGraphInfomaxAnomalE(encoder_net).to(device)
    
    optimizer = optim.Adam(dgi_model.parameters(), lr=0.001)
    bce_loss = nn.BCEWithLogitsLoss()
    
    dgi_model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        salted_feats = edge_feats.clone()
        if dgi_model.training:
            salted_feats += torch.randn_like(salted_feats) * 0.02
            
        pos_scores, neg_scores = dgi_model(edge_index, salted_feats)
        pos_loss = bce_loss(pos_scores[train_mask], torch.ones_like(pos_scores[train_mask]))
        neg_loss = bce_loss(neg_scores[train_mask], torch.zeros_like(neg_scores[train_mask]))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        
    encoder_net.eval()
    with torch.no_grad():
        train_embeddings = encoder_net(edge_index, edge_feats).cpu().numpy()[train_mask]
        
    calculated_contamination = float(np.sum(y_train == 1) / len(y_train))
    calculated_contamination = max(0.01, min(0.49, calculated_contamination))
    
    downstream_detector = get_anomal_e_detector(contamination=calculated_contamination)
    downstream_detector.fit(train_embeddings)
    
    anomal_e_system = FullAnomalEPipeline(encoder_net, downstream_detector)
    anomal_e_system.save_pipeline()
    
    fresh_encoder = EGraphSAGEEncoder(graph_data['num_nodes'], X_train.shape[1], embedding_dim)
    evaluation_ready_system = FullAnomalEPipeline.load_pipeline(fresh_encoder, device=device)
    
    raw_decisions = evaluation_ready_system.predict(edge_index, edge_feats, mask=test_mask)
    binary_preds = np.where(raw_decisions == 1, 0, 1) # Map Isolation Forest outcome (-1/1) to (1/0)
    
    print("\n[Evaluation Summary] True Unsupervised Binary Anomal-E Outcomes:")
    evaluate_and_save("Anomal-E (Full Pipeline)", y_test, binary_preds)

if __name__ == "__main__":
    main()