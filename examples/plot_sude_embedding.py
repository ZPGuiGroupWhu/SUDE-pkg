from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from sude import sude


data_path = Path(__file__).resolve().parents[1] / "benchmarks" / "rice.csv"
data = np.loadtxt(data_path, delimiter=",")

m = data.shape[1]
X = data[:, : m - 1]
labels = data[:, m - 1]

start_time = time.time()
embedding = sude(X, k1=0)
elapsed = time.time() - start_time

print(f"Elapsed time: {elapsed:.3f}s")

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=4)
plt.title("SUDE embedding on rice benchmark")
plt.show()
