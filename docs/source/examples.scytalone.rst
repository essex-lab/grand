Example 2: Scytalone Dehydratase
================================

When the system contains a non-standard small molecule, such as a protein-bound ligand, a few extra steps are necessary.
The ``grand.utils.write_conect()`` function write CONECT lines to a PDB file for the ligand bonds, which is necessary for OpenMM to understand the ligand topology from a PDB structure.
Additionally, an XML file for the ligand parameters should be written using the ``grand.utils.create_ligand_xml()`` function.
These functions could be run prior to the simulation script if desired.

The documentation for these functions can be found in the "grand package" section.

.. literalinclude:: ../../examples/scytalone/scytalone.py
    :language: python
