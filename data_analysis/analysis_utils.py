from PyWGCNA import wgcna
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from astropy.stats import biweight_midcovariance

def compute_biweight_correlation(exprs):
    """
    Compute Biweight Midcorrelation using R's WGCNA package.
    :param exprs: Pandas DataFrame containing gene expression data.
    :return: Numpy array of the Biweight Midcorrelation matrix.
    """
    
    # Define an R script and pass it to R through `rpy2`
    robjects.r('''
    # Load the WGCNA package in R
    library(WGCNA)

    # Define an R function to compute Biweight Midcorrelation
    biweight_midcorrelation <- function(exprs) {
        # Convert input dataframe to a matrix
        cor_matrix <- bicor(as.matrix(exprs))

        # Convert the result back to a dataframe
        return(as.data.frame(cor_matrix))
    }
    ''')

    # Activate conversion between Pandas DataFrames and R DataFrames
    pandas2ri.activate()
    
    # Convert Python Pandas DataFrame to R DataFrame
    r_df = pandas2ri.py2rpy(exprs)
    
    # Call the R function
    r_func = robjects.globalenv['biweight_midcorrelation']
    r_result = r_func(r_df)
    
    # Convert R DataFrame back to Pandas DataFrame
    correlation_df = pandas2ri.rpy2py(r_result)

    return correlation_df

def compute_biweight_correlation_astropy(exprs):
    """
    Computes the Biweight Midcorrelation using the Astropy library.
    
    :param exprs: Numpy array or sparse matrix (samples x genes).
    :return: Numpy array representing the biweight midcorrelation matrix.
    """
    if not isinstance(exprs, np.ndarray):
        exprs = exprs.toarray()  # Convert sparse matrix to dense array
        
    # Compute the biweight midcovariance matrix
    bicov_matrix = biweight_midcovariance(exprs.T)  # Transpose so variables are rows

    # Convert covariance matrix to correlation matrix
    diag_sqrt = np.sqrt(np.diag(bicov_matrix))  # Standard deviations
    biweight_corr_matrix = bicov_matrix / np.outer(diag_sqrt, diag_sqrt)  # Normalize

    return biweight_corr_matrix

def pickSoftThreshold(exprs, corrType="biweight", networkType="unsigned", **kwargs):
    """
    Wrapper for WGCNA's pickSoftThreshold method to include Biweight Midcorrelation and Spearman correlation.

    :param exprs: Pandas DataFrame with gene expression data.
    :param corrType: Correlation method ('biweight', 'spearman', 'pearson').
    :param networkType: Network type ('unsigned', 'signed', 'signed hybrid').
    :return: Soft-thresholding power and data summary.
    """

    correlation_methods = {
        "biweight": lambda x: compute_biweight_correlation(x),
        "spearman": lambda x: pd.DataFrame(stats.spearmanr(x, axis=0)[0]),
        "pearson": lambda x: x  # Use raw expression data
    }

    # Validate corrType
    if corrType not in correlation_methods:
        raise ValueError(f"Invalid corrType '{corrType}'. Choose from {list(correlation_methods.keys())}.")
    
    # Compute the correlation
    correlation = correlation_methods[corrType](exprs)

    # If Pearson, use raw exprs data
    if corrType == "pearson":
        power, sft = wgcna.WGCNA.pickSoftThreshold(adjacency, dataIsExpr=True, networkType=networkType, **kwargs)
    else:
        # Convert Biweight & Spearman correlation into adjacency based on networkType
        if networkType == "unsigned":
            adjacency = np.abs(correlation)
        elif networkType == "signed":
            adjacency = (1 + correlation) / 2
        elif networkType == "signed hybrid":
            adjacency = correlation.copy()
            adjacency[adjacency < 0] = 0
        else:
            raise ValueError(f"Invalid networkType '{networkType}'. Choose from ['unsigned', 'signed', 'signed hybrid'].")

        # Run pickSoftThreshold() on the precomputed adjacency matrix (dataIsExpr=False) or raw data (dataIsExpr=True)
        power, sft = wgcna.WGCNA.pickSoftThreshold(adjacency, dataIsExpr=False, networkType=networkType, **kwargs)

    return power, sft

def adjacency(exprs, corrType="pearson", networkType="unsigned"):
    """
    Computes the adjacency matrix for gene co-expression networks based on correlation and network type.

    :param exprs: Pandas DataFrame with gene expression data (genes as columns).
    :param corrType: Correlation type - "pearson", "spearman", or "biweight".
    :param networkType: Network type - "unsigned", "signed", or "signed hybrid".
    :return: Adjacency matrix (numpy array).
    """
    
    correlation_methods = {
        "pearson": lambda x: np.corrcoef(x.T),
        "spearman": lambda x: stats.spearmanr(x, axis=0)[0],
        "biweight": compute_biweight_correlation
    }
    
    # Compute correlation 
    correlation = correlation_methods[corrType](exprs)
    
    # Compute adjacency based on networkType
    if networkType == "unsigned":
        adjacency = np.abs(correlation)
    elif networkType == "signed":
        adjacency = (1 + correlation) / 2
    elif networkType == "signed hybrid":
        adjacency = np.maximum(0, correlation)
    else:
        raise ValueError(f"Invalid networkType '{networkType}'.")

    return adjacency

def scaleFreeFitIndex(connectivity, output_path, nBreaks=10):
    """
    Evaluates the scale-free topology fit and generates related plots.

    :param connectivity: Array-like or Series containing node connectivity values.
    :param output_path: Path where the plots will be saved.
    :param nBreaks: Number of bins for discretizing connectivity (default=10).
    
    :return: DataFrame with calculated fit indices, fitted models, predictions, and bin edges.
    """

    # Create a DataFrame to store connectivity values
    df = pd.DataFrame({'data': connectivity})

    # Discretize connectivity values into bins
    df['discretized_k'] = pd.cut(df['data'], nBreaks)

    # Compute mean connectivity for each bin
    dk = df.groupby('discretized_k')['data'].mean().reset_index()
    dk.columns = ['discretized_k', 'dk']

    # Compute probability of each bin
    p_dk = df['discretized_k'].value_counts(normalize=True).reset_index()
    p_dk.columns = ['discretized_k', 'p_dk']

    # Define bin edges
    bin_edges = np.linspace(start=df['data'].min(), stop=df['data'].max(), num=nBreaks + 1)

    # Compute bin centers for missing values imputation
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Merge mean connectivity (dk) and probability (p_dk) into df
    df = pd.merge(dk, p_dk, on='discretized_k', how='outer')

    # Handle missing values in `dk` by replacing only where necessary
    if df['dk'].isnull().any():
        df.loc[df['dk'].isnull(), 'dk'] = bin_centers[:df['dk'].isnull().sum()]
    
    # Handle missing values in `p_dk`
    df['p_dk'].fillna(0, inplace=True)

    # Compute log-transformed variables
    df['log_dk'] = np.log10(df['dk'].replace(0, np.nan))  # Avoid log(0)
    df['log_p_dk'] = np.log10(df['p_dk'].replace(0, np.nan) + 1e-9)
    df['log_p_dk_10'] = np.power(10, df['log_dk'])

    # Fit linear models to evaluate scale-free topology
    model1 = ols(formula='log_p_dk ~ log_dk', data=df.dropna()).fit()
    model2 = ols(formula='log_p_dk ~ log_dk + log_p_dk_10', data=df.dropna()).fit()
    # Compute predictions from the models
    predict1 = model1.predict(df["log_dk"])
    predict2 = df["log_p_dk_10"] * model2.params[2] + df["log_dk"] * model2.params[1] + model2.params[0]
            
    # Generate and save Scale-Free Topology plot
    plt.figure()
    plt.scatter(df["log_dk"], df["log_p_dk"], s=10, label="Data")
    plt.plot(df["log_dk"], predict1, color="black", linewidth=1, label="Linear Fit")
    plt.plot(df["log_dk"], predict2, color="red", linewidth=1, label="Truncated Fit")
    plt.xlabel("log10(k_mean)")
    plt.ylabel("log10(p(r))")
    plt.title(f"Scale-Free Topology: R²={model1.rsquared:.3f}, slope={model1.params[1]:.3f}, trunc. R²={model2.rsquared_adj:.3f}")
    plt.legend()
    plt.savefig(f"{output_path}/figures/scale_free_topology.png")
    plt.close()

    # Generate and save Frequency Distribution of Connectivity plot
    bin_widths = np.diff(bin_edges)
    plt.figure()
    plt.bar(bin_edges[:-1], df['p_dk'].values * 100, width=bin_widths, align='edge', color='red', edgecolor='black', hatch='////')
    plt.xlabel("Connectivity k")
    plt.ylabel("Frequency p(r) (%)")
    plt.title("Frequency Distribution of Connectivity")
    plt.savefig(f"{output_path}/figures/frequency_distribution.png")
    plt.close()

    # Return results
    return df, (model1, model2), (predict1, predict2), bin_edges

