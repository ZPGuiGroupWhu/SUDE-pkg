User guide
==========

The public API is the ``sude`` function:

.. code-block:: python

   from sude import sude

   embedding = sude(X, no_dims=2, k1=20)

Key parameters
--------------

``no_dims``
   Output embedding dimension.

``k1``
   Number of nearest neighbours used for landmark sampling. Set ``k1=0`` to
   disable landmark sampling.

``normalize``
   Apply min-max scaling before the embedding is computed.

``large``
   Switch to the blockwise optimization path for large datasets.
