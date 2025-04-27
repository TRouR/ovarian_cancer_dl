# HGSOC Stage Prediction

This section focuses on the training, optimization, and evaluation of machine learning models for predicting the stage of High-Grade Serous Ovarian Carcinomas (HGSOC) ( 2 classes: "high" vs "low").

---

## Dataset Description
- The processed dataset after the WGCNA analysis consists of:
  - **3679 samples**
  - **513 gene expression profiles**
- Data splitting:
  - 85% Training set
  - 7.5% Validation set
  - 7.5% Test set
- Class distribution is preserved across all subsets.
- Gene expression profiles were standardized:
  - Training set normalized (mean=0, std=1).
  - Validation and test sets normalized using training set statistics.

---

## Optimization Strategy
- **Optuna** was used for hyperparameter optimization, leveraging the **Tree-Structured Parzen Estimator (TPE)**.
- The validation set was used during optimization, while the test set was reserved strictly for final evaluation.

---

## Models Trained
The following models were optimized and compared:

| Model           | Key Features |
|:----------------|:-------------|
| **SVM-rbf**      | Radial Basis Function kernel; tuned C and gamma parameters |
| **XGBoost**      | Linear booster; tuned regularization parameters and subsample ratios |
| **Neural Network** | 5 fully connected layers; batch normalization, dropout, ReLU activations |
| **GCNN-GAT**     | Graph Attention Network; tuned thresholds, heads, hidden dimensions |

The input for all models consists of the same 513 gene features.

---

## Key Model Hyperparameters

- **SVM-rbf**:
  - C = 8.780
  - Gamma = 4.7e-4

- **XGBoost**:
  - Booster: linear
  - Lambda (L2) = 1.09e-2
  - Alpha (L1) = 1.1857e-6
  - Subsample = 0.3658
  - Colsample_bytree = 0.7154
  - n_rounds = 50

- **Neural Network**:
  - Hidden Layers: [171, 110, 425, 518, 243]
  - Dropout: 0.4458
  - Activation: ReLU
  - Learning Rate: 2.04e-4
  - L2 Regularization (λ): 3.54e-4
  - L1 Regularization (λ): 3.0381e-2
  - Epochs: 7
  - Batch Size: 64

- **GCNN-GAT**:
  - Threshold on adjacency matrix: 5.4687e-8
  - GAT Layers: 2
  - Hidden Dimensions: ~10
  - Heads per layer: ~2
  - Fully Connected Layers: [434, 491, 31, 508, 295]
  - Dropout: 0.1876 (GAT layers), 0.3555 (FC layers)
  - Activation: Elu
  - Attention Activation: LeakyReLU (slope=0.25)
  - Learning Rate: 9.6e-4
  - L2 Regularization (λ): 2.05e-4
  - L1 Regularization (λ): 4.5e-4
  - Epochs: 18
  - Batch Size: 64

---

## Performance Results

| Model | Validation Accuracy | Validation F1 | Test Accuracy | Test F1 |
|:------|:---------------------|:--------------|:--------------|:-------|
| SVM-rbf        | 0.8691 | 0.6817 | 0.8545 | 0.6024 |
| XGBoost        | 0.8618 | 0.6812 | 0.8582 | 0.6379 |
| Neural Network | 0.8800 | 0.7394 | 0.8618 | 0.6640 |
| GCNN-GAT       | **0.8836** | **0.7557** | **0.8655** | **0.7008** |

✅ The GCNN-GAT model achieved the best overall performance both on validation and test sets.

---

## Feature Importance and Biological Insights

**Saliency scores** were computed from the fully connected part of the GCNN-GAT model to identify the most important genes for predicting the "high" stage of HGSOC.

| Top genes from F1 | Top genes from F2 | Top genes from F3 |
|:------------------|:------------------|:------------------|
| VSIG4             | ALG10B            | KCNMB3            |
| **PIK3CA**        | **PBX1**           | APBB2             |
| COL11A1           | EPB41L3           | GAPDH             |
| C19orf12          | CRISPLD2          | KRT6              |
| FABP4             | MROH5             | LOC81691          |

- **VSIG4**: Implicated in immune evasion and tumor progression. Higher expression correlates with advanced stage ovarian cancers.
- **PIK3CA**: Activates PI3K/AKT/mTOR signaling pathway; known for oncogenic mutations in HGSOC.
- **PBX1**: Regulated by NOTCH signaling; promotes cell proliferation and carcinogenesis in ovarian cancer.

Thus, the GAT model identifies biologically meaningful genes that can serve as potential biomarkers or therapeutic targets.

---

## Notes

- The adjacency matrix for the GCNN-GAT model was computed **only** on the training set to avoid information leakage.
- Class distributions were strictly preserved across all splits.

---

