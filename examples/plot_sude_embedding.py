from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from sude import SUDE


data_path = Path(__file__).resolve().parents[1] / "benchmarks" / "rice.csv"
data = np.loadtxt(data_path, delimiter=",")

m = data.shape[1]
X = data[:, : m - 1]
labels = data[:, m - 1]

start_time = time.time()
embedding = SUDE(
    n_components=2,
    n_neighbors=0,
    init="pca",
    max_iter=50,
).fit_transform(X)
elapsed = time.time() - start_time

print(f"Elapsed time: {elapsed:.3f}s")

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=4)
plt.title("SUDE embedding on rice benchmark")
plt.show()
