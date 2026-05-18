User guide
==========

SUDE 0.2.0 uses optimized probability construction, gradient computation, and
batched non-landmark embedding. Numba-accelerated kernels are installed by
default and are enabled automatically for datasets that are large enough to
benefit from them.

The primary API now follows the scikit-learn estimator pattern:

.. code-block:: python

   from sude import SUDE

   model = SUDE(n_components=2, n_neighbors=20, init="le")
   embedding = model.fit_transform(X)
   new_embedding = model.transform(X_new)

The function wrapper uses the same sklearn-style parameter names:

.. code-block:: python

   from sude import sude

   embedding = sude(X, n_components=2, n_neighbors=20, init="le")

Key parameters
--------------

``n_components``
   Output embedding dimension. This corresponds to ``no_dims`` in the
   original function interface.

``n_neighbors``
   Number of nearest neighbours used for landmark sampling. This corresponds
   to ``k1`` in the paper. Set ``n_neighbors=0`` to disable landmark sampling.

``init``
   Initialization method. This corresponds to ``initialize`` in the original
   function interface.

``max_iter``
   Maximum number of optimization epochs. This corresponds to ``T_epoch`` in
   the paper.

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
