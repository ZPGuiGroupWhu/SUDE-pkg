User guide
==========

SUDE 0.2.0 uses optimized probability construction, gradient computation, and
batched non-landmark embedding. Numba-accelerated kernels are installed by
default and are enabled automatically for datasets that are large enough to
benefit from them.

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

Numba acceleration
------------------

``NUMBA_AVAILABLE`` indicates whether numba is installed. The automatic policy
is controlled by two module-level thresholds in ``sude.learning``:

.. code-block:: python

   import sude.learning as sude_learning

   sude_learning.NUMBA_AUTO_MIN_SAMPLES = 3000
   sude_learning.NUMBA_AUTO_MIN_LANDMARKS = 512

SUDE uses numba only when both the unique input sample count and the landmark
count meet these thresholds. You can set different positive integers before
calling ``fit`` or ``sude``. If either threshold is invalid, SUDE falls back to
using numba whenever numba is installed.
