"""
Description
-----------
This file contains functions written to test the functions in the grand.potential sub-module

Marley Samways
"""

import os
import unittest
import numpy as np
from simtk.unit import *
from simtk.openmm.app import *
from simtk.openmm import *
from grand import potential
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'potential')


class TestPotential(unittest.TestCase):
    """
    Class to store the tests for grand.potential
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

        return None

    def test_get_lambdas(self):
        """
        Test the get_lambda_values() function, designed to retrieve steric and
        electrostatic lambda values from a single lambda value
        """
        # Test several lambda values between 0 and 1 - should interpolate linearly
        assert all(np.isclose(potential.get_lambda_values(1.00), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(0.75), [1.0, 0.5]))
        assert all(np.isclose(potential.get_lambda_values(0.50), [1.0, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(0.25), [0.5, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(0.00), [0.0, 0.0]))
        # Test behaviour outside of these limits - should stay within 0 and 1
        assert all(np.isclose(potential.get_lambda_values(2.00), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(1.50), [1.0, 1.0]))
        assert all(np.isclose(potential.get_lambda_values(-0.50), [0.0, 0.0]))
        assert all(np.isclose(potential.get_lambda_values(-1.00), [0.0, 0.0]))

        return None

    def test_calc_mu(self):
        """
        Test that the calc_mu function performs sensibly
        """
        # Need to set up a system first

        # Load a pre-equilibrated water box
        pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water_box-eq.pdb')))

        # Set up system
        ff = ForceField("tip3p.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12.0 * angstroms,
                                 constraints=HBonds, switchDistance=10 * angstroms)

        # Run free energy calculation using grand
        log_file = os.path.join(outdir, 'free_energy_test.log')
        free_energy = potential.calc_mu_ex(system=system, topology=pdb.topology, positions=pdb.positions,
                                           box_vectors=pdb.topology.getPeriodicBoxVectors(), temperature=298*kelvin,
                                           n_lambdas=6, n_samples=5, n_equil=1,
                                           log_file=log_file)

        # Check that a free energy has been returned
        # Make sure that the returned value has units
        assert isinstance(free_energy, Quantity)
        # Make sure that the value has units of energy
        assert free_energy.unit.is_compatible(kilocalorie_per_mole)

        return None

    def test_calc_std_volume(self):
        """
        Test that the calc_std_volume function performs sensibly
        """
        # Need to set up a system first

        # Load a pre-equilibrated water box
        pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water_box-eq.pdb')))

        # Set up system
        ff = ForceField("tip3p.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12.0 * angstroms,
                                 constraints=HBonds, switchDistance=10 * angstroms)

        # Run std volume calculation using grand
        std_volume = potential.calc_std_volume(system=system, topology=pdb.topology, positions=pdb.positions,
                                               box_vectors=pdb.topology.getPeriodicBoxVectors(),
                                               temperature=298*kelvin, n_samples=10, n_equil=1)

        # Check that a volume has been returned
        # Make sure that the returned value has units
        assert isinstance(std_volume, Quantity)
        # Make sure that the value has units of volume
        assert std_volume.unit.is_compatible(angstroms**3)

        return None
