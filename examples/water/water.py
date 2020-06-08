"""
Description
-----------
Example script of how to run GCMC/MD in OpenMM for a simple water system

Note that this simulation is only an example, and is not long enough
to see equilibrated behaviour

Marley Samways
"""

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import numpy as np

from openmmtools.integrators import BAOABIntegrator

import grand

# Load in a water box PDB...
pdb = PDBFile('water_box-eq.pdb')

# Add ghost waters,
pdb.topology, pdb.positions, ghosts = grand.utils.add_ghosts(pdb.topology,
                                                             pdb.positions,
                                                             n=10,
                                                             pdb='water-ghosts.pdb')

ff = ForceField('tip3p.xml')
system = ff.createSystem(pdb.topology,
                         nonbondedMethod=PME,
                         nonbondedCutoff=12.0*angstroms,
                         switchDistance=10.0*angstroms,
                         constraints=HBonds)

# Create GCMC sampler object - sampling the entire simulation volume
gcmc_mover = grand.samplers.StandardGCMCSystemSampler(system=system,
                                                      topology=pdb.topology,
                                                      temperature=300*kelvin,
                                                      boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
                                                      log='water-gcmc.log',
                                                      dcd='water-raw.dcd',
                                                      rst='water-gcmc.rst7',
                                                      overwrite=False)

# Langevin integrator
integrator = BAOABIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)

platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Switch off ghost waters and in sphere
gcmc_mover.initialise(simulation.context, ghosts)

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

# Run simulation
print("\n\nProduction")
for i in range(50):
    # Carry out 100 GCMC moves per 1 ps of MD
    simulation.step(500)
    gcmc_mover.move(simulation.context, 100)
    # Write data out
    gcmc_mover.report(simulation)

#
# Need to process the trajectory for visualisation
#

# Move ghost waters away
grand.utils.shift_ghost_waters(ghost_file='gcmc-ghost-wats.txt',
                               topology='water-ghosts.pdb',
                               trajectory='water-raw.dcd',
                               output='water-gcmc.dcd')
