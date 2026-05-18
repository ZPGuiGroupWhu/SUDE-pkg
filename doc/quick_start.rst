Quick start
===========

Install the package in editable mode from the project root:

.. code-block:: bash

   uv run python -m pip install -e .

Numba-accelerated kernels are installed by default with the package. They are
enabled automatically when the unique input sample count is at least ``3000``
and the landmark count is at least ``512``.

You can adjust these module-level thresholds before fitting:

.. code-block:: python

   import sude.learning as sude_learning

   sude_learning.NUMBA_AUTO_MIN_SAMPLES = 5000
   sude_learning.NUMBA_AUTO_MIN_LANDMARKS = 1024

Both values must be positive integers. If either value is invalid, SUDE uses
numba whenever numba is installed.

Run the package tests:

.. code-block:: bash

   uv run python -m unittest discover -s tests

Run the example script:

.. code-block:: bash

   uv run python examples/plot_sude_embedding.py

Quick estimator example:

.. code-block:: python

   import numpy as np
   from sude import SUDE

   X = np.loadtxt("benchmarks/rice.csv", delimiter=",")[:, :-1]
   embedding = SUDE(n_components=2, n_neighbors=10, init="le").fit_transform(X)
