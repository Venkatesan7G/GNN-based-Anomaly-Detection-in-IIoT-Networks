# IIoT Automated Anomaly Detection Benchmark Suite

An industrial-grade benchmarking framework for comparing classic supervised classifiers, deep sequence models, and self-supervised Graph Neural Networks (GNNs) on Industrial Internet of Things (IIoT) network telemetry.

---

## 🚀 Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Benchmark

### 1. In-Domain Training & Evaluation (WUSTL-IIoT-2021)

Place the **WUSTL-IIoT-2021** dataset at:

```text
data/wustl_iiot_2021.csv
```

Run the main training pipeline:

```bash
python main.py
```

This will:

- Train all benchmark models on the WUSTL-IIoT-2021 dataset.
- Evaluate their in-domain performance.
- Save trained model weights to the `models/` directory.

---

### 2. Cross-Dataset Generalization (Edge-IIoTset)

Evaluate the trained models on the **Edge-IIoTset** dataset without retraining or preprocessing:

```bash
python cross_validate.py
```

This benchmark measures how well each model generalizes to an entirely different industrial network environment.

---

# 📥 Generated Files

Running the benchmark creates the following directories:

## `hybrid_dataset/`

If the raw Edge-IIoTset dataset is not found locally, the framework automatically downloads or generates a structurally compatible 50,000-sample subset:

```text
Edge-IIoTset_Selected_Dataset.csv
```

---

## `models/`

Contains serialized model weights, including:

### Anomal-E

- `anomal_e_encoder.pt`
- `anomal_e_detector.pkl`

### Deep Learning Baselines

- CNN PyTorch checkpoints
- LSTM PyTorch checkpoints

### Classical Machine Learning

- Decision Tree (`joblib`)
- Random Forest (`joblib`)

---

## `hybrid_result/`

Stores generated evaluation figures, including:

```text
cross_dataset_evaluation.png
```

along with additional benchmark visualizations.

---

# 🎯 Benchmark Objective

The purpose of this project is to compare conventional supervised learning methods with a self-supervised graph-based anomaly detector under both:

- **In-domain evaluation**
- **Cross-domain generalization**

The benchmark demonstrates how different learning paradigms behave when deployed in previously unseen industrial environments.

---

# 🧠 Architecture Discussion

## Supervised Shortcut Learning

Traditional supervised models—including:

- Decision Tree
- Random Forest
- CNN
- LSTM

achieve near-perfect performance (≈1.0 F1-score) on the **WUSTL-IIoT-2021** dataset.

However, these models primarily learn environment-specific statistical patterns rather than a generalized concept of malicious behavior. Examples include:

- Static timing characteristics
- IP address structures
- Simulator-specific artifacts
- Dataset-specific feature distributions

As a result, their performance degrades significantly when evaluated on a different network environment.

---

## Self-Supervised Topological Learning

The **Anomal-E** pipeline uses **Deep Graph Infomax (DGI)** to train a Graph Neural Network without relying on labeled attack data.

Instead of memorizing feature distributions, it learns:

- Communication topology
- Node relationships
- Structural interactions
- Dynamic network behavior

This enables the model to capture relational properties that are more consistent across industrial environments.

---

# 🌍 Cross-Dataset Generalization

When transferring from **WUSTL-IIoT-2021** to **Edge-IIoTset**:

### Supervised Models

- Large performance degradation
- Learned statistical shortcuts no longer exist
- Poor transferability across domains

### Anomal-E

- Learns structural communication patterns instead of flat feature signatures
- Maintains significantly stronger cross-domain performance
- Requires no retraining or dataset-specific preprocessing

---

# 📊 Results

The benchmark exports high-resolution evaluation figures to:

```text
hybrid_result/
```

Example:

```text
cross_dataset_evaluation.png
```

These visualizations compare model performance across both in-domain and cross-dataset evaluations.

---

# 📁 Project Structure

```text
.
├── data/
│   └── wustl_iiot_2021.csv
├── hybrid_dataset/
├── hybrid_result/
├── models/
├── main.py
├── cross_validate.py
├── requirements.txt
└── README.md
```