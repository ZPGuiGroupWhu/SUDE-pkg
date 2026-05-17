User guide
==========

SUDE 0.2.0 uses optimized probability construction, gradient computation, and
batched non-landmark embedding. Numba-accelerated kernels are installed by
default.

The primary API now follows the scikit-learn estimator pattern:

.. code-block:: python

   from sude import SUDE

   model = SUDE(n_components=2, n_neighbors=20, init="spectral")
   embedding = model.fit_transform(X)
   new_embedding = model.transform(X_new)

The legacy function wrapper is still available:

.. code-block:: python

   from sude import sude

   embedding = sude(X, no_dims=2, k1=20, initialize="le")

Key parameters
--------------

``n_components``
   Output embedding dimension.

``n_neighbors``
   Number of nearest neighbours used for landmark sampling. Set ``k1=0`` in
   the legacy function or ``n_neighbors=0`` in the estimator to disable
   landmark sampling.

``normalize``
   Apply min-max scaling before the embedding is computed.

``large``
   Switch to the blockwise optimization path for large datasets.
