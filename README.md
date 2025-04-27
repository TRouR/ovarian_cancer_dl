# **Ovarian Cancer Deep Learning (HGSOC Classification)**

# Project Overview

This study aims to construct a robust model for classifying High-Grade Serous Ovarian Carcinomas (HGSOC) with a particular emphasis on leveraging deep learning techniques.

Given a dataset of **3,814 bulk-rna samples** across **513 gene targets**, the study leverages:
- A **novel Graph Neural Network (GNN)** architecture, tailored specifically for processing graph-structured datasets.
- **Weighted Gene Coexpression Network Analysis (WGCNA)** to capture the nuanced gene interactions.

Each sample is represented as a **graph** where:
- **Nodes** corresponding to genes.  
- **Adjacency matrix 𝐴** is computed using WGCNA to ensure precise information propagation within the graph structure.
  The matrix captures the connections between nodes by assigning **weights** to the correlation between expression levels.
- **Scale-free topology** property is integrated into the adjacency matrix.
- **𝛢** undergoes a bisection process based on an optimized threshold, resulting in the creation of matrix 𝛦 that encodes the neighbours of each node.

Diverse **graph convolutional layers** with **attention** mechanisms are then used to encode local **gene-level** features into higher-order 
co-functional **pathway-level** features. These are then integrated into **fully connected layers** for final classification.

Compared to alternative models, our approach achieves **superior performance** in predicting HGSOC stages.
Furthermore, to augment **interpretability** and glean insights into underlying **biological mechanisms**, we explore
an innovative **full-gradient graph saliency** mechanism.

This approach enables:
- Discernment of **input gene influence** in predicting the actual patient class.
- Interpretation of model decisions, uncovering **pathway-level biomarkers**, pivotal for fathoming disease progression in HGSOC.

---

# 📂 Project Structure

```plaintext
ovarian_cancer_dl/
│
├── data_analysis/
│   ├── data/              # Raw input datasets
│   ├── figures/           # Plots and figures generated during analysis
│   ├── output_data/       # Preprocessed and intermediate data
│   ├── analysis_utils.py  # Utility functions for data processing (e.g., WGCNA computation)
│   └── data_analysis.py   # Scripts for exploratory data analysis
│
├── machine_learning/
│   ├── gat_model.py        # Graph Attention Network (GAT) model architecture
│   ├── gat_optimization.py # Hyperparameter optimization for GAT model
│   ├── nn_model.py         # Fully connected neural network (FCNN) model
│   ├── nn_optimization.py  # Hyperparameter optimization for FCNN model
│   ├── svm_rbf_model.py    # Support Vector Machine (SVM) model with RBF kernel
│   ├── xgboost_model.py    # XGBoost model 
│   ├── utils.py            # General utility functions (loading data, plotting, etc.)
│   └── utils_gat.py        # Specialized utilities for GAT model (training, evaluation, gradient computation)
│
├── LICENSE               
└── README.md
```

# Key Features

- **WGCNA-based** construction of gene interaction networks.
- **Graph Neural Network (GNN)** based modeling with custom architecture.
- **Automatic hyperparameter optimization** using **Optuna**.
- **Deep interpretability** through full-gradient feature attribution.
- Supports traditional **ML baselines** like **SVM** and **XGBoost** for performance comparison.

# 📦 Installation

Install required packages via `pip`:

```bash
pip install -r requirements.txt
```
or via `conda`:
```bash
conda install --file requirements.txt
```

# 🚀 How to Run

1.
2. Run optimizations:

For GAT Model:

```bash
python machine_learning/gat_optimization.py
```

For FCNN:

```bash
python machine_learning/nn_optimization.py
```

Evaluate models and visualize results in the generated directories.




