# Sampling-enabled scalable manifold learning unveils the discriminative cluster structure of high-dimensional data (SUDE)

We propose a scalable manifold learning (SUDE) method that can cope with
large-scale and high-dimensional data in an efficient manner. It starts by
seeking a set of landmarks to construct the low-dimensional skeleton of the
entire data, and then incorporates the non-landmarks into this skeleton based
on the constrained locally linear embedding.

This repository provides the Python version of SUDE. Version 0.2.0 keeps the
public API of the original `sude` package while improving the runtime of the
probability construction, gradient computation, and non-landmark embedding
steps. The MATLAB version can be found at https://github.com/ZPGuiGroupWhu/sude.
The related paper has been published in *Nature Machine Intelligence*:
https://www.nature.com/articles/s42256-025-01112-9.

![image](https://raw.githubusercontent.com/ZPGuiGroupWhu/SUDE-pkg/refs/heads/main/image/sude.jpg)

## Project layout

The project now follows the structure of the
`scikit-learn-contrib/project-template`:

```text
.
|-- .github/workflows/
|-- benchmarks/
|-- doc/
|-- examples/
|-- image/
|-- sude/
|   |-- __init__.py
|   |-- _learning_utils.py
|   |-- _numba_kernels.py
|   |-- _sude.py
|   |-- _version.py
|   |-- learning.py
|   `-- tests/
|-- pyproject.toml
`-- README.md
```

## Installation
Supported `python` versions are `3.8` and above.

This project has been uploaded to [PyPI](https://pypi.org/project/sude/), supporting direct download and installation from pypi

```
pip install sude
```

Numba-accelerated kernels are installed by default.

### Manual installation

```
git clone https://github.com/ZPGuiGroupWhu/SUDE-pkg.git
cd SUDE-pkg
pip install -e .
```

## How to run

The package now exposes both a scikit-learn style estimator class and the
legacy function wrapper.

### Estimator interface

```python
import numpy as np
from sude import SUDE
import time
import matplotlib.pyplot as plt

# Input data
data = np.loadtxt("benchmarks/rice.csv", delimiter=",")

# Obtain data size and true annotations
m = data.shape[1]
X = data[:, :m - 1]
ref = data[:, m - 1]

# Fit a scikit-learn style estimator
start_time = time.time()
model = SUDE(
    n_components=2,
    n_neighbors=10,
    init="pca",
    max_iter=50,
)
Y = model.fit_transform(X)
end_time = time.time()
print("Elapsed time:", end_time - start_time, 's')

plt.scatter(Y[:, 0], Y[:, 1], c=ref, cmap='tab10', s=4)
plt.show()
```

The estimator provides the familiar API:

```python
model = SUDE(n_components=2, n_neighbors=10, init="spectral")
Y_train = model.fit_transform(X_train)
Y_test = model.transform(X_test)
```

### Function interface

The original function entry point remains available for backwards
compatibility:

```python
from sude import sude

Y = sude(X, no_dims=2, k1=10, initialize="le", T_epoch=50)
```

Run the packaged example with:

```bash
uv run python examples/plot_sude_embedding.py
```

Run the test suite with:

```bash
uv run python -m unittest discover -s tests
```

## Citation request
Peng, D., Gui, Z., Wei, W. et al. Sampling-enabled scalable manifold learning unveils the discriminative cluster structure of high-dimensional data. Nat. Mach. Intell. (2025). https://doi.org/10.1038/s42256-025-01112-9


## License
SUDE is released under the MIT License.
