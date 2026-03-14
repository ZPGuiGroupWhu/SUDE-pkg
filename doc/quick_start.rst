Quick start
===========

Install the package in editable mode from the project root:

.. code-block:: bash

   uv run python -m pip install -e .

Run the package tests:

.. code-block:: bash

   uv run python -m unittest discover -s sude/tests

Run the example script:

.. code-block:: bash

   uv run python examples/plot_sude_embedding.py

Quick estimator example:

.. code-block:: python

   import numpy as np
   from sude import SUDE

   X = np.loadtxt("benchmarks/rice.csv", delimiter=",")[:, :-1]
   embedding = SUDE(n_components=2, n_neighbors=10, init="pca").fit_transform(X)
