"""
INDEPENDENT MULTI-CHART GENERATION MODULE
=========================================
Description:
    Parses compiled system metrics from the raw CSV data and exports 
    three separate high-resolution visualization charts using pure 
    matplotlib to eliminate external environment dependency issues.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_individual_plots(csv_path="results/raw_metrics.csv", output_dir="results"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Ensure main.py runs completely first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    models = df['Model'].unique()
    attack_classes = df['Attack Class'].unique()
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    model_colors = {model: colors[i % len(colors)] for i, model in enumerate(models)}

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_indexes = range(len(attack_classes))
        total_models = len(models)
        bar_width = 0.6 / total_models
        
        for i, model in enumerate(models):
            model_df = df[df['Model'] == model]
            
            metric_values = []
            for atk in attack_classes:
                val_series = model_df[model_df['Attack Class'] == atk][metric]
                metric_values.append(val_series.values[0] if not val_series.empty else 0.0)
                
            offsets = [x + (i - total_models/2 + 0.5) * bar_width for x in x_indexes]
            ax.bar(offsets, metric_values, width=bar_width, label=model, color=model_colors[model], edgecolor='black', linewidth=0.5)

        ax.set_title(f'Comparative Binary Performance Analysis: {metric}', fontsize=12, weight='bold', pad=15)
        ax.set_xlabel('System Condition Class', fontsize=11, labelpad=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_ylim(0, 1.1)
        
        ax.set_xticks(x_indexes)
        ax.set_xticklabels(attack_classes, rotation=0, fontsize=10)
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax.set_axisbelow(True)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)
        
        output_file = os.path.join(output_dir, f"{metric.lower().replace('-', '_')}_comparison.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--> Successfully generated individual asset plot at '{output_file}'.")

if __name__ == "__main__":
    generate_individual_plots()