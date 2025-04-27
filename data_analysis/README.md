# Data Analysis

This folder contains the preprocessing and network analysis scripts for the High-Grade Serous Ovarian Carcinomas (HGSOC) gene expression data.  
The goal is to construct a weighted gene co-expression network and compute the adjacency matrix used for the Graph Neural Network (GNN) model.

---

## Scripts Overview

### 1. `data_analysis.py`
- Loads and cleans the gene expression (`GSE135820_exprs.csv`), feature (`GSE135820_features.csv`), and phenotype (`GSE135820_pheno.csv`) data.
- Preprocesses the data:
  - Removes duplicate samples and features.
  - Filters only HGSOC patients with known stage.
  - Updates WGCNA's AnnData object with gene and sample metadata.
- Performs network construction:
  - Computes biweight midcorrelation between genes.
  - Constructs weighted unsinged adjacency matrix.
  - Determines the optimal soft-thresholding power (β) ensuring scale-free topology.
- Outputs processed data for machine learning and generates figures analyzing network properties.

### 2. `analysis_utils.py`
- Utility functions for correlation computation, adjacency matrix construction, and evaluation of network scale-free topology.
- Important functions:
  - `compute_biweight_correlation()`
  - `pickSoftThreshold()`
  - `adjacency()`
  - `scaleFreeFitIndex()`
- Enables flexible choice of correlation type: **Pearson**, **Spearman**, or **Biweight**.

---

## Figures Generated

### Figure 1: Hierarchical Clustering of Samples
![Hierarchical Clustering](../figures/Figure%201.png)  
Hierarchical clustering based on Euclidean distance. A threshold of 45 was set to exclude 135 obvious sample outliers (colored purple).

---

### Figure 2: Scale-Free Topology and Mean Connectivity Analysis
![Scale-Free Topology and Mean Connectivity](../figures/Figure%202.png)  
- **Left:** Scale-free topology fit index (R²) across different powers.
- **Right:** Mean connectivity for the same powers.

---

### Figure 3: Frequency Distribution of Connectivity
![Frequency Distribution](../figures/Figure%203.png)  
Shows that most genes have low connectivity, consistent with a scale-free network property.

---

### Figure 4: Scale-Free Topology Linearity Check
![Scale-Free Linearity](../figures/Figure%204.png)  
Log-log plot demonstrating the linear relation between log(p(r)) and log(k_mean), with R² ≈ 0.967 for the linear fit and R² ≈ 0.991 for the truncated fit.

---

### Figure 5: Node Connectivity in Gene Network
![Node Connectivity](../figures/Figure%205.png)  
Each point represents the number of connections (k) of a gene, confirming the presence of a few highly connected hub genes.

---

## Preprocessing Summary

- **Missing Data Handling:** Samples with extremely low weights are treated as missing. Genes and samples with zero variance were excluded.
- **Outlier Removal:** Samples visibly distant in hierarchical clustering are removed.
- **Adjacency Matrix Construction:** Biweight midcorrelation is used to calculate the weighted adjacency matrix.
- **Scale-Free Topology:** Power β=6 selected based on R² > 0.8 criterion for scale-free network structure.
- **Interpretation:** The network exhibits typical scale-free properties — few highly connected nodes (hub genes) and many lowly connected genes.

---

## Outputs

- **`output_data/`** folder:
  - `exprs.csv`: Processed gene expression matrix.
  - `traits.csv`: Corresponding phenotype metadata.
  - `adjacency matrix`: Saved for GNN training.
- **`figures/`** folder:
  - Figures shown above to support analysis and preprocessing decisions.

---

## Notes
- Biweight midcorrelation is computed using **R's WGCNA** package through **rpy2**.
- Extensive care is taken to ensure a biologically meaningful, interpretable, and robust gene co-expression network.

---

