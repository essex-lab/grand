Example 3: Bulk Water
=====================

This example is slightly different in that bulk water is sampled over the entire system volume.
This is not an efficient use of GCMC for water sampling, but can sometimes be useful for testing purposes.
Note the use of the ``StandardGCMCSystemSampler`` object, rather than ``StandardGCMCSphereSampler``.

.. literalinclude:: ../../examples/water/water.py
    :language: python
