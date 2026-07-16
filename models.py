"""
NEURAL FRAMEWORK & STRUCTURAL GRAPH ENCODER CONFIGURATIONS (BUG FIXES INCLUDED)
==============================================================================
Description:
    Defines tabular and deep neural classifiers, the E-GraphSAGE encoder,
    and a stabilized DeepGraphInfomax self-supervised system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
import joblib

def get_tabular_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=15),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
    }

class CNNModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1) # [Batch, Channels=1, Width]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1) # [Batch, Seq_len=1, Input_dim]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class EGraphSAGEEncoder(nn.Module):
    def __init__(self, num_nodes, edge_feat_dim, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.edge_linear = nn.Linear(edge_feat_dim, embedding_dim)
        self.w_self = nn.Linear(embedding_dim, embedding_dim)
        self.w_neigh = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, edge_index, edge_features):
        device = edge_index.device
        num_nodes = self.node_embeddings.num_embeddings
        
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        edge_repr = F.relu(self.edge_linear(edge_features))
        src_emb = self.node_embeddings(src_nodes)
        
        msg = src_emb + edge_repr
        
        agg_neigh = torch.zeros(num_nodes, self.embedding_dim, device=device)
        agg_neigh.index_add_(0, dst_nodes, msg)
        
        h_nodes = F.relu(self.w_self(self.node_embeddings.weight) + self.w_neigh(agg_neigh))
        h_nodes = self.dropout(h_nodes)
        
        edge_embeddings = h_nodes[src_nodes] * h_nodes[dst_nodes]
        return edge_embeddings

class DeepGraphInfomaxAnomalE(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.discriminator = nn.Bilinear(encoder.embedding_dim, encoder.embedding_dim, 1)
        
    def forward(self, edge_index, edge_features, corrupted_features=None):
        pos_embeddings = self.encoder(edge_index, edge_features)
        
        if corrupted_features is None:
            perm = torch.randperm(edge_features.size(0))
            corrupted_features = edge_features[perm]
            
        neg_embeddings = self.encoder(edge_index, corrupted_features)
        
        summary = torch.sigmoid(torch.mean(pos_embeddings, dim=0, keepdim=True))
        
        # FIX: Changed from expand_like (which does not exist) to expand_as
        summary_expanded = summary.expand_as(pos_embeddings)
        
        pos_scores = self.discriminator(pos_embeddings, summary_expanded)
        neg_scores = self.discriminator(neg_embeddings, summary_expanded)
        
        return pos_scores, neg_scores

def get_anomal_e_detector(contamination=0.05):
    return IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

class FullAnomalEPipeline:
    def __init__(self, encoder, detector=None):
        self.encoder = encoder
        self.detector = detector
        
    def save_pipeline(self, encoder_path="models/anomal_e_encoder.pt", detector_path="models/anomal_e_detector.pkl"):
        torch.save(self.encoder.state_dict(), encoder_path)
        if self.detector is not None:
            joblib.dump(self.detector, detector_path)
            
    @classmethod
    def load_pipeline(cls, encoder_net, device, encoder_path="models/anomal_e_encoder.pt", detector_path="models/anomal_e_detector.pkl"):
        encoder_net.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        encoder_net.to(device)
        encoder_net.eval()
        detector = joblib.load(detector_path)
        return cls(encoder_net, detector)
        
    def predict(self, edge_index, edge_features, mask=None):
        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder(edge_index, edge_features).cpu().numpy()
            
        if mask is not None:
            embeddings = embeddings[mask]
            
        return self.detector.predict(embeddings)