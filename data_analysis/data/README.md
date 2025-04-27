# High-Grade Serous Ovarian Carcinoma Data (GSE135820)

This dataset (GSE135820) contains expression profiles from 4077 samples, including 3829 unique clinical samples of high-grade serous ovarian cancer (HGSOC), 
with additional biological and technical replicates. Expression was measured using a custom NanoString panel of 513 mRNA probes, including housekeeping genes, 
across three sites: Vancouver, Los Angeles, and Melbourne.

For the WGCNA analysis, only the 4041 HGSOC-diagnosed samples were retained. Samples with unknown cancer stage (227 cases) were excluded, yielding a final dataset of 3814 samples and 513 genes. 

The raw count data were processed by applying a log2 transformation, subtracting the average expression of five housekeeping genes (ACTB, RPL19, SDHA, POLR1B, PGK1), 
and normalizing against reference samples.

For more details, see [GSE135820 on NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135820).
