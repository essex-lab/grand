"""
Description
-----------
This file contains functions written to test the functions in the grand.samplers sub-module

Marley Samways
"""

import os
import unittest
import numpy as np
from copy import deepcopy
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from grand import samplers
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'samplers')


def setup_BaseGrandCanonicalMonteCarloSampler():
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global base_gcmc_sampler
    global base_gcmc_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                             constraints=HBonds)

    base_gcmc_sampler = samplers.BaseGrandCanonicalMonteCarloSampler(system=system, topology=pdb.topology,
                                                            temperature=300*kelvin,
                                                            ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                            log=os.path.join(outdir, 'basegcmcsampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    base_gcmc_simulation = Simulation(pdb.topology, system, integrator, platform)
    base_gcmc_simulation.context.setPositions(pdb.positions)
    base_gcmc_simulation.context.setVelocitiesToTemperature(300*kelvin)
    base_gcmc_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    base_gcmc_sampler.context = base_gcmc_simulation.context

    return None


def setup_GCMCSphereSampler():
    """
    Set up variables for the GCMCSphereSampler
    """
    # Make variables global so that they can be used
    global gcmc_sphere_sampler
    global gcmc_sphere_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                 {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

    gcmc_sphere_sampler = samplers.GCMCSphereSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                          referenceAtoms=ref_atoms, sphereRadius=4*angstroms,
                                          ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                          log=os.path.join(outdir, 'gcmcspheresampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    gcmc_sphere_simulation = Simulation(pdb.topology, system, integrator, platform)
    gcmc_sphere_simulation.context.setPositions(pdb.positions)
    gcmc_sphere_simulation.context.setVelocitiesToTemperature(300*kelvin)
    gcmc_sphere_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    gcmc_sphere_sampler.initialise(gcmc_sphere_simulation.context, [3054, 3055, 3056, 3057, 3058])

    return None


def setup_StandardGCMCSphereSampler():
    """
    Set up variables for the StandardGCMCSphereSampler
    """
    # Make variables global so that they can be used
    global std_gcmc_sphere_sampler
    global std_gcmc_sphere_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                 {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

    std_gcmc_sphere_sampler = samplers.StandardGCMCSphereSampler(system=system, topology=pdb.topology,
                                                                 temperature=300*kelvin, referenceAtoms=ref_atoms,
                                                                 sphereRadius=4*angstroms,
                                                                 ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                                 log=os.path.join(outdir, 'stdgcmcspheresampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    std_gcmc_sphere_simulation = Simulation(pdb.topology, system, integrator, platform)
    std_gcmc_sphere_simulation.context.setPositions(pdb.positions)
    std_gcmc_sphere_simulation.context.setVelocitiesToTemperature(300*kelvin)
    std_gcmc_sphere_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    std_gcmc_sphere_sampler.initialise(std_gcmc_sphere_simulation.context, [3054, 3055, 3056, 3057, 3058])

    return None


def setup_NonequilibriumGCMCSphereSampler():
    """
    Set up variables for the GrandCanonicalMonteCarloSampler
    """
    # Make variables global so that they can be used
    global neq_gcmc_sphere_sampler
    global neq_gcmc_sphere_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
    ff = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                 {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)

    neq_gcmc_sphere_sampler = samplers.NonequilibriumGCMCSphereSampler(system=system, topology=pdb.topology,
                                                                       temperature=300*kelvin, referenceAtoms=ref_atoms,
                                                                       sphereRadius=4*angstroms,
                                                                       integrator=integrator,
                                                                       nPropStepsPerPert=10, nPertSteps=1,
                                                                       ghostFile=os.path.join(outdir, 'bpti-ghost-wats.txt'),
                                                                       log=os.path.join(outdir, 'neqgcmcspheresampler.log'))

    # Define a simulation
    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    neq_gcmc_sphere_simulation = Simulation(pdb.topology, system, neq_gcmc_sphere_sampler.compound_integrator, platform)
    neq_gcmc_sphere_simulation.context.setPositions(pdb.positions)
    neq_gcmc_sphere_simulation.context.setVelocitiesToTemperature(300*kelvin)
    neq_gcmc_sphere_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    neq_gcmc_sphere_sampler.initialise(neq_gcmc_sphere_simulation.context, [3054, 3055, 3056, 3057, 3058])

    return None


def setup_GCMCSystemSampler():
    """
    Set up variables for the GCMCSystemSampler
    """
    # Make variables global so that they can be used
    global gcmc_system_sampler
    global gcmc_system_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water-ghosts.pdb')))
    ff = ForceField('tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    gcmc_system_sampler = samplers.GCMCSystemSampler(system=system, topology=pdb.topology, temperature=300*kelvin,
                                                     boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
                                                     ghostFile=os.path.join(outdir, 'water-ghost-wats.txt'),
                                                     log=os.path.join(outdir, 'gcmcsystemsampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    gcmc_system_simulation = Simulation(pdb.topology, system, integrator, platform)
    gcmc_system_simulation.context.setPositions(pdb.positions)
    gcmc_system_simulation.context.setVelocitiesToTemperature(300*kelvin)
    gcmc_system_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    gcmc_system_sampler.initialise(gcmc_system_simulation.context, [2094, 2095, 2096, 2097, 2098])

    return None


def setup_StandardGCMCSystemSampler():
    """
    Set up variables for the StandardGCMCSystemSampler
    """
    # Make variables global so that they can be used
    global std_gcmc_system_sampler
    global std_gcmc_system_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water-ghosts.pdb')))
    ff = ForceField('tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    std_gcmc_system_sampler = samplers.StandardGCMCSystemSampler(system=system, topology=pdb.topology,
                                                                 temperature=300*kelvin,
                                                                 boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
                                                                 ghostFile=os.path.join(outdir, 'water-ghost-wats.txt'),
                                                                 log=os.path.join(outdir, 'stdgcmcsystemsampler.log'))

    # Define a simulation
    integrator = LangevinIntegrator(300 * kelvin, 1.0/picosecond, 0.002*picoseconds)

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    std_gcmc_system_simulation = Simulation(pdb.topology, system, integrator, platform)
    std_gcmc_system_simulation.context.setPositions(pdb.positions)
    std_gcmc_system_simulation.context.setVelocitiesToTemperature(300*kelvin)
    std_gcmc_system_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    std_gcmc_system_sampler.initialise(std_gcmc_system_simulation.context, [2094, 2095, 2096, 2097, 2098])

    return None


def setup_NonequilibriumGCMCSystemSampler():
    """
    Set up variables for the StandardGCMCSystemSampler
    """
    # Make variables global so that they can be used
    global neq_gcmc_system_sampler
    global neq_gcmc_system_simulation

    pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water-ghosts.pdb')))
    ff = ForceField('tip3p.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12 * angstroms,
                              constraints=HBonds)

    integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)

    neq_gcmc_system_sampler = samplers.NonequilibriumGCMCSystemSampler(system=system, topology=pdb.topology,
                                                                       temperature=300*kelvin, integrator=integrator,
                                                                       boxVectors=np.array(pdb.topology.getPeriodicBoxVectors()),
                                                                       ghostFile=os.path.join(outdir,
                                                                                              'water-ghost-wats.txt'),
                                                                       log=os.path.join(outdir,
                                                                                        'neqgcmcsystemsampler.log'))

    # Define a simulation

    try:
        platform = Platform.getPlatformByName('CUDA')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
        except:
            platform = Platform.getPlatformByName('CPU')

    neq_gcmc_system_simulation = Simulation(pdb.topology, system, neq_gcmc_system_sampler.compound_integrator, platform)
    neq_gcmc_system_simulation.context.setPositions(pdb.positions)
    neq_gcmc_system_simulation.context.setVelocitiesToTemperature(300*kelvin)
    neq_gcmc_system_simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

    # Set up the sampler
    neq_gcmc_system_sampler.initialise(neq_gcmc_system_simulation.context, [2094, 2095, 2096, 2097, 2098])

    return None


class TestBaseGrandCanonicalMonteCarloSampler(unittest.TestCase):
    """
    Class to store the tests for the GrandCanonicalMonteCarloSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_BaseGrandCanonicalMonteCarloSampler()

        return None

    def test_move(self):
        """
        Make sure the GrandCanonicalMonteCarloSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: base_gcmc_sampler.move(base_gcmc_simulation.context))

        return None

    def test_report(self):
        """
        Make sure the BaseGrandCanonicalMonteCarloSampler.report() method works correctly
        """
        # Delete some ghost waters so they can be written out
        ghosts = [3054, 3055, 3056, 3057, 3058]
        base_gcmc_sampler.deleteGhostWaters(ghostResids=ghosts)

        # Report
        base_gcmc_sampler.report(base_gcmc_simulation)

        # Check the output to the ghost file
        assert os.path.isfile(os.path.join(outdir, 'bpti-ghost-wats.txt'))
        # Read which ghosts were written
        with open(os.path.join(outdir, 'bpti-ghost-wats.txt'), 'r') as f:
            n_lines = 0
            lines = f.readlines()
            for line in lines:
                if len(line.split()) > 0:
                    n_lines += 1
        assert n_lines == 1
        ghosts_read = [int(resid) for resid in lines[0].split(',')]
        assert all(np.isclose(ghosts, ghosts_read))

        return None

    def test_reset(self):
        """
        Make sure the BaseGrandCanonicalMonteCarloSampler.reset() method works correctly
        """
        # Set tracked variables to some non-zero values
        base_gcmc_sampler.n_accepted = 1
        base_gcmc_sampler.n_moves = 1
        base_gcmc_sampler.Ns = [1]

        # Reset base_gcmc_sampler
        base_gcmc_sampler.reset()

        # Check that the values have been reset
        assert base_gcmc_sampler.n_accepted == 0
        assert base_gcmc_sampler.n_moves == 0
        assert len(base_gcmc_sampler.Ns) == 0

        return None


class TestGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the GCMCSphereSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_GCMCSphereSampler()

        return None

    def test_initialise(self):
        """
        Make sure the GCMCSphereSampler.initialise() method works correctly
        """

        # Make sure the variables are all updated
        assert isinstance(gcmc_sphere_sampler.context, Context)
        assert isinstance(gcmc_sphere_sampler.positions, Quantity)
        assert isinstance(gcmc_sphere_sampler.sphere_centre, Quantity)

        return None

    def test_deleteWatersInGCMCSphere(self):
        """
        Make sure the GCMCSphereSampler.deleteWatersInGCMCSphere() method works correctly
        """
        # Now delete the waters in the sphere
        gcmc_sphere_sampler.deleteWatersInGCMCSphere()
        new_ghosts = gcmc_sphere_sampler.getWaterStatusResids(0)
        # Check that the list of ghosts is correct
        assert new_ghosts == [70, 71, 3054, 3055, 3056, 3057, 3058]
        # Check that the variables match there being no waters in the GCMC region
        assert gcmc_sphere_sampler.N == 0
        assert all([x in [0, 2] for x in gcmc_sphere_sampler.water_status.values()])

        return None

    def test_updateGCMCSphere(self):
        """
        Make sure the GCMCSphereSampler.updateGCMCSphere() method works correctly
        """
        # Get initial gcmc_resids and status
        gcmc_waters = deepcopy(gcmc_sphere_sampler.getWaterStatusResids(1))
        sphere_centre = deepcopy(gcmc_sphere_sampler.sphere_centre)
        N = gcmc_sphere_sampler.N

        # Update the GCMC sphere (shouldn't change as the system won't have moved)
        state = gcmc_sphere_simulation.context.getState(getPositions=True, getVelocities=True)
        gcmc_sphere_sampler.updateGCMCSphere(state)

        # Make sure that these values are all still the same
        assert all(np.isclose(gcmc_waters, gcmc_sphere_sampler.getWaterStatusResids(1)))
        assert all(np.isclose(sphere_centre._value, gcmc_sphere_sampler.sphere_centre._value))
        assert N == gcmc_sphere_sampler.N

        return None

    def test_move(self):
        """
        Make sure the GCMCSphereSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: gcmc_sphere_sampler.move(gcmc_sphere_simulation.context))

        return None

    def test_insertRandomWater(self):
        """
        Make sure the GCMCSphereSampler.insertRandomWater() method works correctly
        """
        # Insert a random water
        new_positions, wat_id, atom_ids = gcmc_sphere_sampler.insertRandomWater()

        # Check that the indices returned are integers - may not be type int
        assert wat_id == int(wat_id)
        assert all([i == int(i) for i in atom_ids])

        # Check that the new positions are different to the old positions
        assert all([any([new_positions[i][j] != gcmc_sphere_sampler.positions[i][j] for j in range(3)])
                    for i in atom_ids])
        assert all([all([new_positions[i][j] == gcmc_sphere_sampler.positions[i][j] for j in range(3)])
                    for i in range(len(new_positions)) if i not in atom_ids])

        return None

    def test_deleteRandomWater(self):
        """
        Make sure the GCMCSphereSampler.deleteRandomWater() method works correctly
        """
        # Insert a random water
        delete_water, atom_indices = gcmc_sphere_sampler.deleteRandomWater()

        # Check that the indices returned are integers
        assert delete_water == int(delete_water)
        assert all([i == int(i) for i in atom_indices])

        return None


class TestStandardGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the StandardGCMCSphereSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_StandardGCMCSphereSampler()

        return None

    def test_move(self):
        """
        Make sure the StandardGCMCSphereSampler.move() method works correctly
        """
        # Run a handful of GCMC moves
        n_moves = 10
        std_gcmc_sphere_sampler.move(std_gcmc_sphere_simulation.context, n_moves)

        # Check that all of the appropriate variables seem to have been updated
        # Hard to test individual moves as they are rarely accepted - just need to check the overall behaviour
        assert std_gcmc_sphere_sampler.n_moves == n_moves
        assert 0 <= std_gcmc_sphere_sampler.n_accepted <= n_moves
        assert len(std_gcmc_sphere_sampler.Ns) == n_moves
        assert len(std_gcmc_sphere_sampler.acceptance_probabilities) == n_moves
        assert isinstance(std_gcmc_sphere_sampler.energy, Quantity)
        assert std_gcmc_sphere_sampler.energy.unit.is_compatible(kilocalories_per_mole)

        return None


class TestNonequilibriumGCMCSphereSampler(unittest.TestCase):
    """
    Class to store the tests for the NonequilibriumGCMCSphereSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_NonequilibriumGCMCSphereSampler()

        return None

    def test_move(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.move() method works correctly
        """
        neq_gcmc_sphere_sampler.reset()

        # Just run one move, as they are a bit more expensive
        neq_gcmc_sphere_sampler.move(neq_gcmc_sphere_simulation.context, 1)

        # Check some of the variables have been updated as appropriate
        assert neq_gcmc_sphere_sampler.n_moves == 1
        assert 0 <= neq_gcmc_sphere_sampler.n_accepted <= 1
        assert len(neq_gcmc_sphere_sampler.Ns) == 1
        assert len(neq_gcmc_sphere_sampler.acceptance_probabilities) == 1

        # Check the NCMC-specific variables
        assert isinstance(neq_gcmc_sphere_sampler.velocities, Quantity)
        assert neq_gcmc_sphere_sampler.velocities.unit.is_compatible(nanometers/picosecond)
        assert len(neq_gcmc_sphere_sampler.insert_works) + len(neq_gcmc_sphere_sampler.delete_works) == 1
        assert 0 <= neq_gcmc_sphere_sampler.n_left_sphere <= 1
        assert 0 <= neq_gcmc_sphere_sampler.n_explosions <= 1

        return None

    def test_insertionMove(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.insertionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_sphere_sampler.context = neq_gcmc_sphere_simulation.context
        state = neq_gcmc_sphere_sampler.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        neq_gcmc_sphere_sampler.positions = deepcopy(state.getPositions(asNumpy=True))
        neq_gcmc_sphere_sampler.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        neq_gcmc_sphere_sampler.updateGCMCSphere(state)

        # Set to NCMC integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_sphere_sampler.insertionMove()

        # Reset the compound integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None

    def test_deletionMove(self):
        """
        Make sure the NonequilibriumGCMCSphereSampler.deletionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_sphere_sampler.context = neq_gcmc_sphere_simulation.context
        state = neq_gcmc_sphere_sampler.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        neq_gcmc_sphere_sampler.positions = deepcopy(state.getPositions(asNumpy=True))
        neq_gcmc_sphere_sampler.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        neq_gcmc_sphere_sampler.updateGCMCSphere(state)

        # Set to NCMC integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_sphere_sampler.deletionMove()

        # Reset the compound integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None


class TestGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the GCMCSystemSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Need to create the sampler
        setup_GCMCSystemSampler()

        return None

    def test_initialise(self):
        """
        Make sure the GCMCSystemSampler.initialise() method works correctly
        """
        # Make sure the variables are all updated
        assert isinstance(gcmc_system_sampler.context, Context)
        assert isinstance(gcmc_system_sampler.positions, Quantity)
        assert isinstance(gcmc_system_sampler.simulation_box, Quantity)

        return None

    def test_move(self):
        """
        Make sure the GCMCSystemSampler.move() method works correctly
        """
        # Shouldn't be able to run a move with this sampler
        self.assertRaises(NotImplementedError, lambda: gcmc_system_sampler.move(gcmc_system_simulation.context))

        return None

    def test_insertRandomWater(self):
        """
        Make sure the GCMCSystemSampler.insertRandomWater() method works correctly
        """
        # Insert a random water
        new_positions, wat_id, atom_ids = gcmc_system_sampler.insertRandomWater()

        # Check that the indices returned are integers - may not be type int
        assert wat_id == int(wat_id)
        assert all([i == int(i) for i in atom_ids])

        # Check that the new positions are different to the old positions
        assert all([any([new_positions[i][j] != gcmc_system_sampler.positions[i][j] for j in range(3)])
                    for i in atom_ids])
        assert all([all([new_positions[i][j] == gcmc_system_sampler.positions[i][j] for j in range(3)])
                    for i in range(len(new_positions)) if i not in atom_ids])

        return None

    def test_deleteRandomWater(self):
        """
        Make sure the GCMCSystemSampler.deleteRandomWater() method works correctly
        """
        # Insert a random water
        delete_water, atom_ids = gcmc_system_sampler.deleteRandomWater()

        # Check that the indices returned are integers
        assert delete_water == int(delete_water)
        assert all([i == int(i) for i in atom_ids])

        return None


class TestStandardGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the StandardGCMCSystemSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_StandardGCMCSystemSampler()

        return None

    def test_move(self):
        """
        Make sure the StandardGCMCSystemSampler.move() method works correctly
        """
        # Run a handful of GCMC moves
        n_moves = 10
        std_gcmc_system_sampler.move(std_gcmc_system_simulation.context, n_moves)

        # Check that all of the appropriate variables seem to have been updated
        # Hard to test individual moves as they are rarely accepted - just need to check the overall behaviour
        assert std_gcmc_system_sampler.n_moves == n_moves
        assert 0 <= std_gcmc_system_sampler.n_accepted <= n_moves
        assert len(std_gcmc_system_sampler.Ns) == n_moves
        assert len(std_gcmc_system_sampler.acceptance_probabilities) == n_moves
        assert isinstance(std_gcmc_system_sampler.energy, Quantity)
        assert std_gcmc_system_sampler.energy.unit.is_compatible(kilocalories_per_mole)

        return None


class TestNonequilibriumGCMCSystemSampler(unittest.TestCase):
    """
    Class to store the tests for the NonequilibriumGCMCSystemSampler class
    """
    @classmethod
    def setUpClass(cls):
        """
        Get things ready to run these tests
        """
        # Make the output directory if needed
        if not os.path.isdir(os.path.join(os.path.dirname(__file__), 'output')):
            os.mkdir(os.path.join(os.path.dirname(__file__), 'output'))
        # Create a new directory if needed
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # If not, then clear any files already in the output directory so that they don't influence tests
        else:
            for file in os.listdir(outdir):
                os.remove(os.path.join(outdir, file))

        # Create sampler
        setup_NonequilibriumGCMCSystemSampler()

        return None

    def test_move(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.move() method works correctly
        """
        neq_gcmc_system_sampler.reset()

        # Just run one move, as they are a bit more expensive
        neq_gcmc_system_sampler.move(neq_gcmc_system_simulation.context, 1)

        # Check some of the variables have been updated as appropriate
        assert neq_gcmc_system_sampler.n_moves == 1
        assert 0 <= neq_gcmc_system_sampler.n_accepted <= 1
        assert len(neq_gcmc_system_sampler.Ns) == 1
        assert len(neq_gcmc_system_sampler.acceptance_probabilities) == 1

        # Check the NCMC-specific variables
        assert isinstance(neq_gcmc_system_sampler.velocities, Quantity)
        assert neq_gcmc_system_sampler.velocities.unit.is_compatible(nanometers/picosecond)
        assert len(neq_gcmc_system_sampler.insert_works) + len(neq_gcmc_system_sampler.delete_works) == 1
        assert 0 <= neq_gcmc_system_sampler.n_explosions <= 1

        return None

    def test_insertionMove(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.insertionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_system_sampler.context = neq_gcmc_system_simulation.context
        state = neq_gcmc_system_sampler.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        neq_gcmc_system_sampler.positions = deepcopy(state.getPositions(asNumpy=True))
        neq_gcmc_system_sampler.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Set to NCMC integrator
        neq_gcmc_system_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_system_sampler.insertionMove()

        # Reset the compound integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None

    def test_deletionMove(self):
        """
        Make sure the NonequilibriumGCMCSystemSampler.deletionMove() method works correctly
        """
        # Prep for a move
        # Read in positions
        neq_gcmc_system_sampler.context = neq_gcmc_system_simulation.context
        state = neq_gcmc_system_sampler.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        neq_gcmc_system_sampler.positions = deepcopy(state.getPositions(asNumpy=True))
        neq_gcmc_system_sampler.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Set to NCMC integrator
        neq_gcmc_system_sampler.compound_integrator.setCurrentIntegrator(1)

        # Just run one move to make sure it doesn't crash
        neq_gcmc_system_sampler.deletionMove()

        # Reset the compound integrator
        neq_gcmc_sphere_sampler.compound_integrator.setCurrentIntegrator(0)

        return None
