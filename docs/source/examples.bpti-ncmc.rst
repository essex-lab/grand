Example 4: Nonequilibrium GCMC Moves
====================================

This is a simple example showing how grand can be used to enhance GCMC moves using NCMC.
Rather than inserting or deleting a water and then immediately evaluating the energy change, here, waters are gradually inserted or deleted according to a nonequilibrium protocol, based on NCMC theory.
The majority of the script below is taken from the previous BPTI example, with a few changes made to demonstrate the differences between conventional GCMC sampling and that enhanced by NCMC.
These differences are summarised below:

- The ``StandardGCMCSphereSampler`` object is replaced with ``NonequilibriumGCMCSphereSampler`` (there is also an equivalent ``NonequilibriumGCMCSystemSampler`` object.
- The Sampler object now takes three additional arguments: the ``integrator`` argument is needed for the propagation steps, ``nPertSteps`` is the number of perturbation steps over which the insertion/deletion is carried out, and ``nPropStepsPerPert`` is the number of propagation/relaxation steps between perturbations (the total number of relaxation steps used over the move will be ``(nPertSteps + 1) * nPropStepsPerPert``)
- The ``Simulation`` object is created using the ``gcmc_mover.compound_integrator`` object.
- Given the improved acceptance rate (and increased cost), fewer GCMC moves are required when using NCMC (note that the full benefit may not be seen for this BPTI system, as once the three waters are inserted, then the likelihood of further acceptances becomes very low).

The documentation for these functions can be found in the "grand package" section.
The full script is included below.

.. literalinclude:: ../../examples/bpti-ncmc/bpti-ncmc.py
    :language: python
