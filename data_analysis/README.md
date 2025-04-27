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
  - Determines the optimal soft-thresholding power $β$ ensuring scale-free topology.
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
![Figure 1](https://github.com/user-attachments/assets/9623bda8-5934-4be3-b00f-97b6387e45d7)
Hierarchical clustering based on Euclidean distance and Average linkage. 
A threshold of 45 was set to exclude 135 obvious sample outliers (colored purple).

---

### Figure 2: Scale-Free Topology and Mean Connectivity Analysis
![Figure 2](https://github.com/user-attachments/assets/b72d9974-b8ba-4d65-947b-55c977b76ddb)
- **Left:** Scale-free topology fit index $R²$ across different powers.
- **Right:** Mean connectivity for the same powers.

---

### Figure 3: Frequency Distribution of Connectivity
![Figure 3](https://github.com/user-attachments/assets/158f46da-445e-4c50-90a8-32c59c0589ca) 

The property of scale-free topology can be realised by a fairly asymmetric frequency distribution of connectivity.

---

### Figure 4: Scale-Free Topology Linearity Check
![Figure 4](https://github.com/user-attachments/assets/2a94dd0e-4157-40c0-abee-16b0c7a104a2) 

Log-log plot demonstrating the linear relation between $log(p(r))$ and $log(k.mean)$, with $R² ≈ 0.967$ for the linear fit and $R² ≈ 0.991$ for the truncated fit.

---

### Figure 5: Node Connectivity in Gene Network
![Figure 5](https://github.com/user-attachments/assets/3d3454e2-0a7b-48b9-8701-642f3beaea01) 

Each point represents the connectivity $k$ of a gene, confirming the presence of a few highly connected hub genes.

---

## Preprocessing Summary

- **Missing Data Handling:**

If weights are provided for the expression values, samples with weights < 0.1 are treated as missing. 
Genes and samples are checkedd for incomplete entries and extremely low variance.

- **Outlier Removal:** Samples visibly distant in hierarchical clustering are removed.
- **Adjacency Matrix Construction:** Biweight midcorrelation is used to calculate the weighted adjacency matrix.
- **Scale-Free Topology:** Power β=6 selected based on $R²$ > 0.8 criterion for scale-free network structure.
- **Interpretation:** The network exhibits typical scale-free properties — few highly connected nodes (hub genes) and many lowly connected genes.

---

## Outputs

- **`output_data/`** folder:
  - `exprs.csv`: Processed gene expression matrix.
  - `traits.csv`: Corresponding phenotype metadata.
- **`figures/`** folder:
  - Figures shown above to support analysis and preprocessing decisions.

---

## Notes

- **Biweight Midcorrelation:**  
  Preferred correlation method, computed using R's WGCNA package via rpy2.
  It is more robust to outliers and better preserves the signal compared to Pearson or Spearman correlations.

- **Network Type — Unsigned:**  
  In an unsigned network, both positive and negative correlations contribute equally to the adjacency.  
  This approach enables the modeling of gene pathways involving both upregulated and downregulated factors, offering a more comprehensive view of regulatory mechanisms.

- **Scale-Free Topology Assessment:**  
  To assess scale-free topology, the degree of linearity between the frequency of connectivity $p(r)$ and the connectivity $r$ in log scale is measured using the linear regression $R²$.  
  Although the connectivity distribution can often be better modeled by an exponentially truncated power law, this method tends to yield high $R²$ values even when the true scale-free   
  property is not strongly satisfied.  
  Therefore, the $R²$ from the simple linear model between $log(p(r))$ and $log(r)$ is preferred for evaluating the topology.

- **Soft-Thresholding Power β:**  
  The parameter β critically affects the network's topology.  
  It should be selected to approximately satisfy the scale-free topology criterion:  
  - Choose β values that lead to a linear regression $R²$ greater than a predefined threshold (typically **0.8**).  
  - Maintain sufficient mean connectivity to avoid overly sparse (uninformative) networks.  
  - A balance between high $R²$ and meaningful connectivity must be achieved.

---
