# Sampling-enabled scalable manifold learning unveils the discriminative cluster structure of high-dimensional data
We propose a scalable manifold learning (SUDE) method that can cope with large-scale and high-dimensional data in an efficient manner. It starts by seeking a set of landmarks to construct the low-dimensional skeleton of the entire data, and then incorporates the non-landmarks into this skeleton based on the constrained locally linear embedding. This project provides the ***Python version of SUDE***. The corresponding paper has been published in ***Nature Machine Intelligence***, and more details can be seen https://www.nature.com/articles/s42256-025-01112-9.

![image](https://github.com/ZPGuiGroupWhu/sude/blob/master/github.png)

# How To Run
Python code of SUDE is in the ```sude_py``` file, where the ```sude``` function provides multiple hyperparameters for user configuration as follows
```python
def sude(
    X,
    no_dims = 2,
    k1 = 20,
    normalize = True,
    large = False,
    initialize = 'le',
    agg_coef = 1.2,
    T_epoch = 50,
):
"""
    This function returns representation of the N by D matrix X in the lower-dimensional space. Each row in X
    represents an observation.

    Parameters are:

    'no_dims'      - A positive integer specifying the number of dimension of the representation Y.
                   Default: 2
    'k1'           - A non-negative integer specifying the number of nearest neighbors for PPS to
                   sample landmarks. It must be smaller than N.
                   Default: adaptive
    'normalize'    - Logical scalar. If true, normalize X using min-max normalization. If features in
                   X are on different scales, 'Normalize' should be set to true because the learning
                   process is based on nearest neighbors and features with large scales can override
                   the contribution of features with small scales.
                   Default: True
    'large'        - Logical scalar. If true, the data can be split into multiple blocks to avoid the problem
                   of memory overflow, and the gradient can be computed block by block using 'learning_l' function.
                   Default: False
    'initialize'   - A string specifying the method for initializing Y before manifold learning.
        'le'       - Laplacian eigenmaps.
        'pca'      - Principal component analysis.
        'mds'      - Multidimensional scaling.
                   Default: 'le'
    'agg_coef'     - A positive scalar specifying the aggregation coefficient.
                   Default: 1.2
    'T_epoch'      - Maximum number of epochs to take.
                   Default: 50
"""
```

The ```main.py``` file provides an example
```python
import pandas as pd
import numpy as np
from sude import sude
import time
import matplotlib.pyplot as plt

# Input data
data = np.array(pd.read_csv('benchmarks/rice.csv', header=None))

# Obtain data size and true annotations
m = data.shape[1]
X = data[:, :m - 1]
ref = data[:, m - 1]

# Perform SUDE embedding
start_time = time.time()
Y = sude(X, k1=10)
end_time = time.time()
print("Elapsed time:", end_time - start_time, 's')

plt.scatter(Y[:, 0], Y[:, 1], c=ref, cmap='tab10', s=4)
plt.show()
```

# Depends
> **scRNA-seq application**

argparse (≥2.0.4), assertthat (≥0.2.1), BiocGenerics (≥0.40.0), BiocSingular (≥1.10.0), ClusterR (≥1.2.5), dotCall64 (≥1.0.1), fields (≥12.5), GenomeInfoDb (≥1.30.1), GenomicRanges (≥1.46.1), geometry (≥0.4.5), ggplot2 (≥3.3.5), grid (≥4.1.0), gtools (≥3.9.2), IRanges (≥2.28.0), MatrixGenerics (≥1.6.0), mclust (≥5.4.7), parallel (≥4.1.0), prodlim (≥2019.11.13), RcppHungarian (≥0.1), readr (≥1.4.0), reshape2 (≥1.4.4), S4Vectors (≥0.30.0), scran (≥1.22.1), scuttle (≥1.4.0), Seurat (≥4.0.5), SingleCellExperiment (≥1.16.0), spam (≥2.7.0), stats4 (≥4.1.0), SummarizedExperiment (≥1.24.0), uwot (≥0.1.10)

Noted: all R packages can be installed from the [CRAN repository](https://cran.r-project.org/) or [Bioconductor](https://www.bioconductor.org/). You can also use the following R scripts to install them all.
```ruby
## Please click Tools->Global Options->Packages, change CRAN repository to a near mirror. Then, execute the following code:
## Install packages from CRAN.
install.packages(c("argparse", "assertthat", "ClusterR", "dotCall64", "fields", "geometry", "ggplot2", "gtools", "mclust", "prodlim", "RcppHungarian", "readr", "reshape2", "Seurat", "spam", "uwot"))
## Determine whether the package "BiocManager" exists, if not, install this package.
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
## Install packages from Bioconductor.
BiocManager::install(c("BiocGenerics", "BiocSingular", "GenomeInfoDb", "GenomicRanges", "IRanges", "MatrixGenerics", "S4Vectors", "scran", "scuttle", "SingleCellExperiment", "SummarizedExperiment"), force = TRUE, update = TRUE, ask = FALSE)
```

> **ECG application**

[Deep Learning Toolbox](https://ww2.mathworks.cn/products/deep-learning.html)

[Signal Processing Toolbox](https://www.mathworks.com/products/signal.html)

# Citation Request
Peng, D., Gui, Z., Wei, W. et al. Sampling-enabled scalable manifold learning unveils the discriminative cluster structure of high-dimensional data. Nat. Mach. Intell. (2025). https://doi.org/10.1038/s42256-025-01112-9


