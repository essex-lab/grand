"""
Description
-----------

Example script for the second phase of a typical equilibration protocol that combines MD sampling and GCMC moves

Marley Samways
Ollie Melling
"""

from simtk.openmm import *
from simtk.openmm.app import *
from simtk.unit import *
from openmmtools.integrators import BAOABIntegrator
from sys import stdout
import numpy as np
import mdtraj
import grand


# Load PDB
pdb = PDBFile('bpti-uvt1.pdb')

# Create system
forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12*angstrom,
                                 switchDistance=10*angstrom, constraints=HBonds)

# Make sure the LJ interactions are being switched
for f in range(system.getNumForces()):
    force = system.getForce(f)
    if 'NonbondedForce' == force.__class__.__name__:
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(1.0*nanometer)
        # Also need to switch off the default long-range LJ, as this isn't used in GCMC/MD
        force.setUseDispersionCorrection(False)

# Define the barostat
system.addForce(MonteCarloBarostat(1*bar, 298*kelvin, 25))

# Define integrator
integrator = BAOABIntegrator(298*kelvin, 1.0/picosecond, 0.002*picosecond)

# Define platform and set precision
platform = Platform.getPlatformByName('CUDA')
platform.setPropertyDefaultValue('Precision', 'mixed')

# Create simulation object
simulation = Simulation(pdb.topology, system, integrator, platform)

# Set positions, velocities and box vectors
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(298*kelvin)
simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

# Add a reporter - write out every 5 ps
simulation.reporters.append(StateDataReporter(stdout, 2500, step=True, time=True, potentialEnergy=True,
                                              temperature=True, volume=True))

# Run for 500 ps
simulation.step(250000)

# Write out a PDB
positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
pdb.topology.setPeriodicBoxVectors(simulation.context.getState(getPositions=True).getPeriodicBoxVectors())
with open('bpti-npt.pdb', 'w') as f:
    PDBFile.writeFile(pdb.topology, positions, f)


