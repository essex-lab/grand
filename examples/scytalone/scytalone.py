"""
Description
-----------
Example script of how to run GCMC/MD in OpenMM for a scytalone dehydratase (SD) system

Note that this simulation is only an example, and is not long enough
to see equilibrated behaviour

Marley Samways
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

from openmmtools.integrators import BAOABIntegrator

import grand

# Write CONECT lines to PDB
grand.utils.write_conect('scytalone-equil.pdb', 'MQ1', 'mq1.prepi', 'sd-conect.pdb')

# Write ligand XML
grand.utils.create_ligand_xml('mq1.prmtop', 'mq1.prepi', 'MQ1', 'mq1.xml')

# Load in PDB file
pdb = PDBFile('sd-conect.pdb')

# Add ghost water molecules, which can be inserted
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology,
                                                             pdb.positions,
                                                             n=5,
                                                             pdb='sd-ghosts.pdb')

# Create system
ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml', 'mq1.xml')
system = ff.createSystem(pdb.topology,
                         nonbondedMethod=PME,
                         nonbondedCutoff=12.0*angstroms,
                         switchDistance=10.0*angstroms,
                         constraints=HBonds)

# Define reference atoms around which the GCMC sphere is based
ref_atoms = [{'name': 'OH', 'resname': 'TYR', 'resid': '23'},
             {'name': 'OH', 'resname': 'TYR', 'resid': '43'}]

# Create GCMC Sampler object
gcmc_mover = grand.samplers.StandardGCMCSphereSampler(system=system,
                                                      topology=pdb.topology,
                                                      temperature=300*kelvin,
                                                      referenceAtoms=ref_atoms,
                                                      sphereRadius=4*angstroms,
                                                      log='sd-gcmc.log',
                                                      dcd='sd-raw.dcd',
                                                      rst='sd-gcmc.rst7',
                                                      overwrite=False)

# BAOAB Langevin integrator (important)
integrator = BAOABIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Switch off ghost waters and in sphere
gcmc_mover.initialise(simulation.context, ghosts)
gcmc_mover.deleteWatersInGCMCSphere()

# Equilibrate water distribution - 10k moves over 5 ps
print("Equilibration...")
for i in range(50):
    # Carry out 200 moves every 100 fs
    gcmc_mover.move(simulation.context, 200)
    simulation.step(50)
print("{}/{} equilibration GCMC moves accepted. N = {}".format(gcmc_mover.n_accepted,
                                                               gcmc_mover.n_moves,
                                                               gcmc_mover.N))

# Add StateDataReporter for production
simulation.reporters.append(StateDataReporter(stdout,
                                              1000,
                                              step=True,
                                              potentialEnergy=True,
                                              temperature=True,
                                              volume=True))
# Reset GCMC statistics
gcmc_mover.reset()

# Run simulation - 5k moves over 50 ps
print("\nProduction")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)

#
# Need to process the trajectory for visualisation
#

# Move ghost waters out of the simulation cell
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt',
                                     topology='sd-ghosts.pdb',
                                     trajectory='sd-raw.dcd')

# Recentre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=23)

# Align the trajectory to the protein
grand.utils.align_traj(t=trj, output='sd-gcmc.dcd')

# Write out a trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=4.0,
                              ref_atoms=ref_atoms,
                              topology='sd-ghosts.pdb',
                              trajectory='sd-gcmc.dcd',
                              output='gcmc_sphere.pdb',
                              initial_frame=True)

