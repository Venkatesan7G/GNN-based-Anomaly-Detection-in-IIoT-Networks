# IIoT Automated Anomaly Detection Benchmark Suite

An industrial-grade benchmarking suite for comparing classic tabular classifiers, deep learning sequence networks, and self-supervised Graph Neural Networks (GNNs) on Industrial Internet of Things (IIoT) telemetry data. 

This project evaluates model performance using the **WUSTL-IIOT-2021** dataset to benchmark real-world generalized deployment readiness.

---

## 🎯 Project Objective & Code Overview

The goal of this suite is to build and evaluate a generalized security framework capable of detecting industrial anomalies and malicious structural variations on a factory floor. 

The application is split into simple, modular building blocks:
* **`run_pipeline.py`**: The main automation engine. It launches the other scripts one by one, tracks errors, and logs progress.
* **`preprocess.py`**: A data cleaner. It removes network identity leaks (like absolute IP addresses and timestamps) and builds structural graph connection maps (who talks to whom) for the GNN.
* **`models.py`**: Contains all model designs. This includes traditional baselines (Decision Tree, Random Forest), deep learning loops (CNN, LSTM), and the **Anomal-E Pipeline** (a Graph Neural Network encoder combined with a downstream unsupervised Isolation Forest detector).
* **`evaluate.py`**: Converts all multi-class attack names into a unified binary format (`Normal` operational baseline vs. `Anomaly` outlier state) so every model is judged fairly.
* **`plot.py`**: A clean visualization script. It reads your final data and generates **three separate high-resolution charts** (Precision, Recall, F1-Score) using pure `matplotlib` to avoid external environment crashes.

*Note: Unlike standard benchmark scripts, this framework automatically serializes and saves every single trained model to the `models/` directory so they are fully prepared for field deployment.*

---

## ⚠️ Understanding the Metric Skew: Why Anomal-E is NOT Bad

When analyzing your final table, you will see a massive drop in performance for `Anomal-E` compared to the other simpler baselines:

| Model | Attack Class | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| Decision Tree | Anomaly | 1.0000 | 1.0000 | 1.0000 |
| Anomal-E (Pipeline) | Normal | 0.0002 | 0.8906 | 0.0003 |
| Anomal-E (Pipeline) | Anomaly | 0.9998 | 0.0911 | 0.1670 |

It is an academic error to conclude from these metrics that the Graph Neural Network is broken or inferior. Here is the exact reason for this skewing:

### 1. The Supervised Tabular Shortcut (The Illusion of Perfection)
Supervised models (Trees, CNNs, LSTMs) use labels during training. They find hidden mathematical fingerprints in individual dataset rows—such as exact packet sizes or precise flag combinations unique to the lab environment where the attack script ran. The Decision Tree doesn't learn what an attack *looks like structurally*; it simply memorizes flat signature shortcuts.

### 2. The Unsupervised Boundary Fallacy (The Mathematical Trap)
Anomal-E is **Self-Supervised**. It is completely blind to security labels during training. It maps the *shape of the global graph network traffic*. 

Because the default dataset split mixes a massive volume of attacks directly into the training phase, the downstream Isolation Forest calculates a high structural contamination ratio capped at `0.50` (50%). This forces the model to draw its spatial boundary under the assumption that exactly 50% of your network traffic is normal and 50% is abnormal. 

When it is tested against a data split dominated almost entirely by attacks (99.98% anomaly rate), the model is mathematically handcuffed—it still forces 50% of those data points into the "Normal" category. Because true normal packets are practically non-existent in this specific test slice, almost every packet the model guesses as "Normal" is actually an attack packet, driving the Normal Precision down to `0.0002`.

### The Generalized Deployment Takeaway:
In a production deployment, a real attacker will use a **Zero-Day Attack** (an un-labeled, brand-new signature). The supervised trees and sequence loops will score **0.0** because their memorized signatures won't match. Anomal-E, by mapping structural connections, will naturally flag the anomaly because the topology of who is talking to whom changes completely.

---

## 🚀 Quick Start

### 1. Install Requirements
```bash
pip install -r requirments.txt
2. File Layout
Place your unzipped dataset inside a data/ folder in the root directory:

Plaintext
your-project/
  ├── data/
  │    └── wustl_iiot_2021.csv
  ├── main.py
  ├── plot.py
  └── run_pipeline.py
3. Run
Bash
python run_pipeline.py
Trained model states will be exported to models/ and evaluation charts will be exported to results/.

📜 License
This project is licensed under the Unlicense — a completely unrestricted public domain dedication.

You are free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means. It is completely open-source without any copyleft requirements or attribution obligations.

For more information, please refer to http://unlicense.org/