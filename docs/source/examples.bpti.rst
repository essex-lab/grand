Example 1: Bovine Pancreatic Trypsin Inhibitor
==============================================

This is a simple example showing how grand can be used to simulate GCMC/MD for a protein solvated in water.
The majority of this script below is composed of standard OpenMM functions, with a few additional parts to execute grand canonical sampling.
These additional functions are described here.

- The ``grand.utils.add_ghosts()`` function is used to add some additional waters to the system topology, so these can be used for GCMC insertion moves - until inserted, these waters are non-interacting.
- The ``ref_atoms`` is used to choose the set of atoms, for which the mean coordinate is used to define the centre of the GCMC sphere - this should be chosen carefully (along with the sphere radius) to ensure the region of interest is well covered.
- The ``grand.samplers.StandardGCMCSphereSampler`` object contains all of the necessary variables for carrying out GCMC moves, and the arguments given should be self-explanatory.
- The ``gcmc_mover.initialise()`` function must be executed before starting the simulation, as this feeds some context-specific variables to ``gcmc_mover`` and ensures that the ghosts are switched off.
- The ``gcmc_mover.deleteWatersInGCMCSphere()`` removes any waters present in the GCMC sphere at the beginning of the simulation, so that the water sampling will be less biased by the initial water locations.
- The ``gcmc_mover.move(simulation.context, 200)`` function executes a number of GCMC moves at a given point. For reasons of efficiency, it is best to carry these out in blocks of at least ~20 moves.
- By running ``gcmc_mover.report()``, a simulation frame is written out and the log file is updated.

The remaining functions are used to process the trajectory for visualisation and analysis:

- ``grand.utils.shift_ghost_waters()`` translates the ghost waters far from the simulation box, so that they will not be confused with interacting waters.
- ``grand.utils.recentre_traj()`` is used to recentre the trajectory on a particular atom. However, this can be expensive, so if this atom does not get close to the edges of the simulation cell, this is not necessary.
- ``grand.utils.align_traj()`` is used to align the protein with respect to the initial frame (or a reference structure, via the ``reference`` argument).
- ``grand.utils.write_sphere_traj`` writes out a PDB trajectory of the GCMC sphere, which may be helpful for visualisation.

The documentation for these functions can be found in the "grand package" section.
The full script is included below.

.. literalinclude:: ../../examples/bpti/bpti.py
    :language: python
