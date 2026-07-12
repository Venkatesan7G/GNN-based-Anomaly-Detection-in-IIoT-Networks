"""
STANDARDIZED INDUSTRIAL METRICS HARNESS
=======================================
Description:
    Computes unified binary metrics (Normal vs Anomaly) across all models 
    to establish a fair, mathematically rigorous benchmark for Anomal-E.
"""

import os
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_and_save(model_name, y_true, y_pred, output_csv="results/raw_metrics.csv"):
    # Unified Binary Schema: 0 is Normal, 1 is Anomaly
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    
    for class_label, metrics in report.items():
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            continue
            
        class_name = "Normal" if str(class_label) == "0" else "Anomaly"
            
        rows.append({
            "Model": model_name,
            "Attack Class": class_name,
            "Precision": round(metrics['precision'], 4),
            "Recall": round(metrics['recall'], 4),
            "F1-Score": round(metrics['f1-score'], 4)
        })
        
    df_new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if os.path.exists(output_csv):
        df_old = pd.read_csv(output_csv)
        df_final = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=['Model', 'Attack Class'], keep='last')
    else:
        df_final = df_new
        
    df_final.to_csv(output_csv, index=False)
    print(df_new.to_string(index=False))