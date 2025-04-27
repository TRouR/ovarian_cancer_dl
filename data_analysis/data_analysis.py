import os
import pandas as pd
import numpy as np
from PyWGCNA import wgcna 
from analysis_utils import *

# Define paths
WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, "data")
OUTPUT_DIR = os.path.join(WORKING_DIR, "output_data")
FIGURES_DIR = os.path.join(WORKING_DIR, "figures")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

############################################################################################
############################################################################################

## Load and process the expression, features, and phenotype data
exprs = pd.read_csv(os.path.join(DATA_DIR, "GSE135820_exprs.csv")).transpose()
features = pd.read_csv(os.path.join(DATA_DIR, "GSE135820_features.csv"))
pheno = pd.read_csv(os.path.join(DATA_DIR, "GSE135820_pheno.csv")).reset_index()

# Prepare phenotype dataframe -> metadata
pheno.set_index(pheno["level_1"], inplace=True)
pheno = pheno.loc[:, "age at diagnosis:ch1":]
pheno.columns = [col.replace(":ch1", "") for col in pheno.columns]

# Filter phenotype data
y = pheno[["Stage", "diagnosis"]].copy()
y = y[(y["Stage"] != "unknown") & (y["diagnosis"] == "HGSOC")].drop(columns="diagnosis")

# Set column names for exprs
exprs.columns = features["ID"].values
exprs = exprs.loc[y.index, :]

# Reset index and rename sample IDs -> required format for the AnnData object of WGCNA
exprs = exprs.reset_index().rename(columns={'level_1': 'sample_id'})

############################################################################################
############################################################################################

## Remove duplicate rows and columns in exprs, and update corresponding y and features data.
# Remove duplicate rows in exprs (Samples).
duplicate_rows = exprs.duplicated()
if duplicate_rows.any():
    print(f"Found {duplicate_rows.sum()} duplicate rows. Removing them...")
    exprs = exprs.loc[~duplicate_rows, :]    
    # Also remove corresponding rows from y
    y = y.iloc[~duplicate_rows.values]

# Step 2: Remove duplicate columns in exprs (Genes).
duplicate_columns = exprs.columns.duplicated()
if duplicate_columns.any():
    print(f"Found {duplicate_columns.sum()} duplicate columns. Removing them...")
    exprs = exprs.loc[:, ~duplicate_columns]
    # Also remove corresponding features from the features DataFrame
    features = features.iloc[~duplicate_columns[1:],]

############################################################################################
############################################################################################

# Create WGCNA object
obj = wgcna.WGCNA(name = "HGSOC",TPMcutoff = 1,
                  RsquaredCut = 0.9, MeanCut = 2,
                  networkType="unsigned", cut = 45,
                  species = "Homo Sapiens",
                  level = "gene", geneExp = exprs, 
                  save = True, outputPath = WORKING_DIR)

# Choose correlation type for the analysis. Options: "biweight", "spearman", "pearson"
obj.correlation = "biweight"

# Add gene info in wgcna anndata object
obj.updateGeneInfo(features[["Customer.Identifier"]], None, ' ', True, 'gene')
# Add metadata in wgcna anndata object
obj.updateMetadata(y,None,' ',True)

# Run preprocessing on wgcna object, including removing obvious genes and samples outliers 
obj.preprocess()

# Save processed data for machine learning 
obj.datExpr.to_df().to_csv(fr"{obj.outputPath}\output_data\exprs.csv")
obj.datExpr.obs.to_csv(fr"{obj.outputPath}\output_data\traits.csv")

# Perform the network topology analysis 
obj.power, obj.sft = pickSoftThreshold(obj.datExpr.to_df(), corrType=obj.correlation,
                                       networkType=obj.networkType,RsquaredCut=obj.RsquaredCut,
                                       MeanCut=obj.MeanCut, powerVector=obj.powers,
                                       nBreaks=10, moreNetworkConcepts=True)

# Find adjacency matrix setting the power
adj = adjacency(obj.datExpr.to_df(), corrType = obj.correlation, networkType = obj.networkType)
adj = adj ** obj.power

# Calculate connectivity
obj.connectivity = np.sum(adj, axis=1) - 1

############################################################################################
############################################################################################

# Run Scale Free Index and generate plots
df, models, predictions, bin_edges = scaleFreeFitIndex(obj.connectivity, obj.outputPath)

############################################################################################
############################################################################################

# Plot Connectivity of Node-Genes in the Network
plt.close()
plt.scatter([*range(obj.datExpr.var.shape[0])],obj.connectivity,s=10)
plt.xlabel(f"Gene Index (from 0 to {obj.datExpr.var.shape[0]})")
plt.ylabel("k",rotation=0)
plt.title("Node Connectivity based on Adjacency Matrix")
plt.savefig(fr"{obj.outputPath}/figures/node-connectivity.png")
plt.close()

############################################################################################
############################################################################################

# Plot Scale Free Topology Fit R^2 and mean connectivity versus powers
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), facecolor='white')
ax[0].plot(obj.sft['Power'], -1 * np.sign(obj.sft['slope']) * obj.sft['SFT.R.sq'], 'o')
for i in range(len(obj.powers)):
    ax[0].text(obj.sft.loc[i, 'Power'],
                -1 * np.sign(obj.sft.loc[i, 'slope']) * obj.sft.loc[i, 'SFT.R.sq'],
                str(obj.sft.loc[i, 'Power']), ha="center", va="center", color='black', weight='bold')
ax[0].axhline(0.8, color='r')
ax[0].set_xlabel("Soft Threshold (power)")
ax[0].set_ylabel(fr"Scale Free Topology Model Fit,{obj.networkType} R^2")
ax[0].title.set_text('Scale independence')

ax[1].plot(obj.sft['Power'], obj.sft['mean(k)'], 'o')
for i in range(len(obj.powers)):
    ax[1].text(obj.sft.loc[i, 'Power'], obj.sft.loc[i, 'mean(k)'],
                str(obj.sft.loc[i, 'Power']), ha="center", va="center", color='r', weight='bold')
ax[1].set_xlabel("Soft Threshold (power)")
ax[1].set_ylabel("Mean Connectivity")
ax[1].title.set_text('Mean connectivity')

fig.tight_layout()
fig.savefig(obj.outputPath + '/figures/summarypower.png')