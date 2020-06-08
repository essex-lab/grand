"""
Description
-----------
Example script of how to run GCMC/MD in OpenMM for a BPTI system

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

# Load in PDB file
pdb = PDBFile('bpti-equil.pdb')

# Add ghost water molecules, which can be inserted
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology,
                                                             pdb.positions,
                                                             n=5,
                                                             pdb='bpti-ghosts.pdb')

# Create system
ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = ff.createSystem(pdb.topology,
                         nonbondedMethod=PME,
                         nonbondedCutoff=12.0*angstroms,
                         switchDistance=10.0*angstroms,
                         constraints=HBonds)

# Define atoms around which the GCMC sphere is based
ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
             {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

gcmc_mover = grand.samplers.StandardGCMCSphereSampler(system=system,
                                                      topology=pdb.topology,
                                                      temperature=300*kelvin,
                                                      referenceAtoms=ref_atoms,
                                                      sphereRadius=4.2*angstroms,
                                                      log='bpti-gcmc.log',
                                                      dcd='bpti-raw.dcd',
                                                      rst='bpti-rst.rst7',
                                                      overwrite=False)

# BAOAB Langevin integrator
integrator = BAOABIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Switch off ghost waters and those in sphere (to start fresh)
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

# Shift ghost waters outside the simulation cell
trj = grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt',
                                     topology='bpti-ghosts.pdb',
                                     trajectory='bpti-raw.dcd')

# Centre the trajectory on a particular residue
trj = grand.utils.recentre_traj(t=trj, resname='TYR', resid=10)

# Align the trajectory to the protein
grand.utils.align_traj(t=trj, output='bpti-gcmc.dcd')

# Write out a PDB trajectory of the GCMC sphere
grand.utils.write_sphere_traj(radius=4.2,
                              ref_atoms=ref_atoms,
                              topology='bpti-ghosts.pdb',
                              trajectory='bpti-gcmc.dcd',
                              output='gcmc_sphere.pdb',
                              initial_frame=True)
