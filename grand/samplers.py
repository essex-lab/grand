# -*- coding: utf-8 -*-

"""
Description
-----------
This module is written to execute GCMC moves with water molecules in OpenMM, via a series of
Sampler objects.

Marley Samways
Ollie Melling
"""

import numpy as np
import mdtraj
import os
import logging
import parmed
import math
from copy import deepcopy
from simtk import unit
from simtk import openmm
from openmmtools.integrators import NonequilibriumLangevinIntegrator

from grand.utils import random_rotation_matrix
from grand.utils import PDBRestartReporter
from grand.potential import get_lambda_values


class BaseGrandCanonicalMonteCarloSampler(object):
    """
    Base class for carrying out GCMC moves in OpenMM.
    All other Sampler objects are derived from this
    """
    def __init__(self, system, topology, temperature, ghostFile="gcmc-ghost-wats.txt", log='gcmc.log',
                 dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Create logging object
        if os.path.isfile(log):
            if overwrite:
                os.remove(log)
            else:
                raise Exception("File {} already exists, not overwriting...".format(log))

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        self.logger.addHandler(file_handler)

        # Set important variables here
        self.system = system
        self.topology = topology
        self.positions = None  # Store no positions upon initialisation
        self.context = None
        self.kT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperature
        self.simulation_box = np.zeros(3) * unit.nanometer  # Set to zero for now

        self.logger.info("kT = {}".format(self.kT.in_units_of(unit.kilocalorie_per_mole)))

        # Find NonbondedForce - needs to be updated to switch waters on/off
        for f in range(system.getNumForces()):
            force = system.getForce(f)
            if force.__class__.__name__ == "NonbondedForce":
                self.nonbonded_force = force
            # Flag an error if not simulating at constant volume
            elif "Barostat" in force.__class__.__name__:
                self.raiseError("GCMC must be used at constant volume - {} cannot be used!".format(force.__class__.__name__))
        
        # Set GCMC-specific variables
        self.N = 0  # Initialise N as zero
        self.Ns = []  # Store all observed values of N
        self.n_moves = 0
        self.n_accepted = 0
        self.acceptance_probabilities = []  # Store acceptance probabilities
        
        # Get parameters for the water model
        self.water_params = self.getWaterParameters("HOH")

        # Get water residue IDs & assign statuses to each
        self.water_resids = self.getWaterResids("HOH")  # All waters
        # Assign each water a status: 0: ghost water, 1: GCMC water, 2: Water not under GCMC tracking (out of sphere)
        self.water_status = np.ones(len(self.water_resids))  # Initially assign all to 1

        # Need to create a customised force to handle softcore steric interactions of water molecules
        # This should prevent any 0/0 energy evaluations
        self.custom_nb_force = None
        self.customiseForces()

        # Need to open the file to store ghost water IDs
        self.ghost_file = ghostFile
        # Check whether to overwrite if the file already exists
        if os.path.isfile(self.ghost_file) and not overwrite:
            self.raiseError("File {} already exists, not overwriting...".format(self.ghost_file))
        else:
            with open(self.ghost_file, 'w') as f:
                pass

        # Store reporters for DCD and restart output
        if dcd is not None:
            # Check whether to overwrite
            if os.path.isfile(dcd):
                if overwrite:
                    # Need to remove before overwriting, so there isn't any mix up
                    os.remove(dcd)
                    self.dcd = mdtraj.reporters.DCDReporter(dcd, 0)
                else:
                    self.raiseError("File {} already exists, not overwriting...".format(dcd))
            else:
                self.dcd = mdtraj.reporters.DCDReporter(dcd, 0)
        else:
            self.dcd = None

        if rst is not None:
            # Check whether to overwrite
            if os.path.isfile(rst) and not overwrite:
                self.raiseError("File {} already exists, not overwriting...".format(rst))
            else:
                # Check whether to use PDB or RST7 for the restart file
                rst_ext = os.path.splitext(rst)[1]
                if rst_ext == '.rst7':
                    self.restart = parmed.openmm.reporters.RestartReporter(rst, 0)
                elif rst_ext == '.pdb':
                    self.restart = PDBRestartReporter(rst, self.topology)
                else:
                    self.raiseError("File extension {} not recognised for restart file".format(rst))
        else:
            self.restart = None

        self.logger.info("BaseGrandCanonicalMonteCarloSampler object initialised")

    def customiseForces(self):
        """
        Create a CustomNonbondedForce to handle water-water interactions and modify the original NonbondedForce
        to ignore water interactions
        """
        #  Need to make sure that the electrostatics are handled using PME (for now)
        if self.nonbonded_force.getNonbondedMethod() != openmm.NonbondedForce.PME:
            self.raiseError("Currently only supporting PME for long range electrostatics")

        # Define the energy expression for the softcore sterics
        energy_expression = ("U;"
                             "U = (lambda^soft_a) * 4 * epsilon * x * (x-1.0);"  # Softcore energy
                             "x = (sigma/reff)^6;"  # Define x as sigma/r(effective)
                             # Calculate effective distance
                             "reff = sigma*((soft_alpha*(1.0-lambda)^soft_b + (r/sigma)^soft_c))^(1/soft_c);"
                             # Define combining rules
                             "sigma = 0.5*(sigma1+sigma2); epsilon = sqrt(epsilon1*epsilon2); lambda = lambda1*lambda2")

        # Create a customised sterics force
        custom_sterics = openmm.CustomNonbondedForce(energy_expression)
        # Add necessary particle parameters
        custom_sterics.addPerParticleParameter("sigma")
        custom_sterics.addPerParticleParameter("epsilon")
        custom_sterics.addPerParticleParameter("lambda")
        # Assume that the system is periodic (for now)
        custom_sterics.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        # Transfer properties from the original force
        custom_sterics.setUseSwitchingFunction(self.nonbonded_force.getUseSwitchingFunction())
        custom_sterics.setCutoffDistance(self.nonbonded_force.getCutoffDistance())
        custom_sterics.setSwitchingDistance(self.nonbonded_force.getSwitchingDistance())
        self.nonbonded_force.setUseDispersionCorrection(False)
        custom_sterics.setUseLongRangeCorrection(self.nonbonded_force.getUseDispersionCorrection())
        # Set softcore parameters
        custom_sterics.addGlobalParameter('soft_alpha', 0.5)
        custom_sterics.addGlobalParameter('soft_a', 1)
        custom_sterics.addGlobalParameter('soft_b', 1)
        custom_sterics.addGlobalParameter('soft_c', 6)

        # Get a list of all water and non-water atom IDs
        water_atom_ids = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid in self.water_resids:
                for atom in residue.atoms():
                    water_atom_ids.append(atom.index)

        # Copy all steric interactions into the custom force, and remove them from the original force
        for atom_idx in range(self.nonbonded_force.getNumParticles()):
            # Get atom parameters
            [charge, sigma, epsilon] = self.nonbonded_force.getParticleParameters(atom_idx)

            # Make sure that sigma is not equal to zero
            if np.isclose(sigma._value, 0.0):
                sigma = 1.0 * unit.angstrom

            # Add particle to the custom force (with lambda=1 for now)
            custom_sterics.addParticle([sigma, epsilon, 1.0])

            # Disable steric interactions in the original force by setting epsilon=0 (keep the charges for PME purposes)
            self.nonbonded_force.setParticleParameters(atom_idx, charge, sigma, abs(0))

        # Copy over all exceptions into the new force as exclusions
        # Exceptions between non-water atoms will be excluded here, and handled by the NonbondedForce
        # If exceptions (other than ignored interactions) are found involving water atoms, we have a problem
        for exception_idx in range(self.nonbonded_force.getNumExceptions()):
            [i, j, chargeprod, sigma, epsilon] = self.nonbonded_force.getExceptionParameters(exception_idx)

            # If epsilon is greater than zero, this is a non-zero exception, which must be checked
            if epsilon > 0.0 * unit.kilojoule_per_mole:
                if i in water_atom_ids or j in water_atom_ids:
                    self.raiseError("Non-zero exception interaction found involving water atoms ({} & {}). grand is"
                                    " not currently able to support this".format(i, j))

            # Add this to the list of exclusions
            custom_sterics.addExclusion(i, j)

        # Add the custom force to the system
        self.system.addForce(custom_sterics)
        self.custom_nb_force = custom_sterics

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []
        
        return None

    def getWaterParameters(self, water_resname="HOH"):
        """
        Get the non-bonded parameters for each of the atoms in the water model used

        Parameters
        ----------
        water_resname : str
            Name of the water residues
    
        Returns
        -------
        wat_params : list
            List of dictionaries containing the charge, sigma and epsilon for each water atom
        """
        wat_params = []  # Store parameters in a list
        for residue in self.topology.residues():
            if residue.name == water_resname:
                for atom in residue.atoms():
                    # Store the parameters of each atom
                    atom_params = self.nonbonded_force.getParticleParameters(atom.index)
                    wat_params.append({'charge' : atom_params[0],
                                       'sigma' : atom_params[1],
                                       'epsilon' : atom_params[2]})
                break  # Don't need to continue past the first instance
        return wat_params

    def getWaterResids(self, water_resname="HOH"):
        """
        Get the residue IDs of all water molecules in the system

        Parameters
        ----------
        water_resname : str
            Name of the water residues

        Returns
        -------
        resid_list : list
            List of residue ID numbers
        """
        resid_list = []
        for resid, residue in enumerate(self.topology.residues()):
            if residue.name == water_resname:
                resid_list.append(resid)
        return resid_list

    def setWaterStatus(self, resid, new_value):
        """
        Set the status of a perticular water to a particular value

        Parameters
        ----------
        resid : int
            Residue to update the status for
        new_value : int
            New value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water
        """
        # Index of this residue
        wat_idx = np.where(np.array(self.water_resids) == resid)[0]
        self.water_status[wat_idx] = new_value
        return None

    def getWaterStatusResids(self, value):
        """
        Get a list of resids which have a particular status value

        Parameters
        ----------
        value : int
            Value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water

        Returns
        -------
        resids : numpy.array
            List of residues which match that status
        """
        resids = [self.water_resids[idx] for idx in np.where(self.water_status == value)[0]]
        return resids

    def getWaterStatusValue(self, resid):
        """
        Get the status value of a particular resid

        Parameters
        ----------
        resid : int
            Residue to update the status for

        Returns
        -------
        value : int
            Value of the water status. 0: ghost, 1: GCMC water, 2: Non-tracked water
        """
        wat_idx = np.where(np.array(self.water_resids) == resid)[0]
        value = self.water_status[wat_idx]
        return value

    def deleteGhostWaters(self, ghostResids=None, ghostFile=None):
        """
        Switch off nonbonded interactions involving the ghost molecules initially added
        This function should be executed before beginning the simulation, to prevent any
        explosions.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        ghostFile : str
            File containing residue IDs of ghost waters. Will switch off those on the
            last line. This will be useful in restarting simulations

        Returns
        -------
        context : simtk.openmm.Context
            Updated context, with ghost waters switched off
        """
        # Get a list of all ghost residue IDs supplied from list and file
        ghost_resids = []
        # Read in list
        if ghostResids is not None:
            for resid in ghostResids:
                ghost_resids.append(resid)

        # Read residues from file if needed
        if ghostFile is not None:
            with open(ghostFile, 'r') as f:
                lines = f.readlines()
                for resid in lines[-1].split(","):
                    ghost_resids.append(int(resid))

        # Switch off the interactions involving ghost waters
        for resid, residue in enumerate(self.topology.residues()):
            if resid in ghost_resids:
                #  Switch off nonbonded interactions involving this water
                atom_ids = []
                for i, atom in enumerate(residue.atoms()):
                    atom_ids.append(atom.index)
                self.adjustSpecificWater(atom_ids, 0.0)
                # Mark that this water has been switched off
                self.setWaterStatus(resid, 0)

        # Calculate N
        self.N = len(self.getWaterStatusResids(1))

        return None

    def adjustSpecificWater(self, atoms, new_lambda):
        """
        Adjust the coupling of a specific water molecule, by adjusting the lambda value

        Parameters
        ----------
        atoms : list
            List of the atom indices of the water to be adjusted
        new_lambda : float
            Value to set lambda to for this particle
        """
        # Get lambda values
        lambda_vdw, lambda_ele = get_lambda_values(new_lambda)

        # Loop over parameters
        for i, atom_idx in enumerate(atoms):
            # Obtain original parameters
            atom_params = self.water_params[i]
            # Update charge in NonbondedForce
            self.nonbonded_force.setParticleParameters(atom_idx,
                                                       charge=(lambda_ele * atom_params["charge"]),
                                                       sigma=atom_params["sigma"],
                                                       epsilon=abs(0.0))
            # Update lambda in CustomNonbondedForce
            self.custom_nb_force.setParticleParameters(atom_idx,
                                                       [atom_params["sigma"], atom_params["epsilon"], lambda_vdw])

        # Update context with new parameters
        self.nonbonded_force.updateParametersInContext(self.context)
        self.custom_nb_force.updateParametersInContext(self.context)
        
        return None

    def report(self, simulation):
        """
        Function to report any useful data

        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            Simulation object being used
        """
        # Get state
        state = simulation.context.getState(getPositions=True, getVelocities=True)

        # Calculate rounded acceptance rate and mean N
        if self.n_moves > 0:
            acc_rate = np.round(self.n_accepted * 100.0 / self.n_moves, 4)
        else:
            acc_rate = np.nan
        mean_N = np.round(np.mean(self.Ns), 4)
        # Print out a line describing the acceptance rate and sampling of N
        msg = "{} move(s) completed ({} accepted ({:.4f} %)). Current N = {}. Average N = {:.3f}".format(self.n_moves,
                                                                                                         self.n_accepted,
                                                                                                         acc_rate,
                                                                                                         self.N,
                                                                                                         mean_N)
        print(msg)
        self.logger.info(msg)

        # Write to the file describing which waters are ghosts through the trajectory
        self.writeGhostWaterResids()

        # Append to the DCD and update the restart file
        if self.dcd is not None:
            self.dcd.report(simulation, state)
        if self.restart is not None:
            self.restart.report(simulation, state)

        return None

    def raiseError(self, error_msg):
        """
        Make it nice and easy to report an error in a consisent way - also easier to manage error handling in future

        Parameters
        ----------
        error_msg : str
            Message describing the error
        """
        # Write to the log file
        self.logger.error(error_msg)
        # Raise an Exception
        self.logger.error(error_msg)

        return None

    def writeGhostWaterResids(self):
        """
        Write out a comma-separated list of the residue IDs of waters which are
        non-interacting, so that they can be removed from visualisations. It is important 
        to execute this function when writing to trajectory files, so that each line
        in the ghost water file corresponds to a frame in the trajectory
        """
        # Need to write this function
        with open(self.ghost_file, 'a') as f:
            ghost_resids = self.getWaterStatusResids(0)
            if len(ghost_resids) > 0:
                f.write("{}".format(ghost_resids[0]))
                if len(ghost_resids) > 1:
                    for resid in ghost_resids[1:]:
                        f.write(",{}".format(resid))
            f.write("\n")

        return None

    def move(self, context, n=1):
        """
        Returns an error if someone attempts to execute a move with the parent object
        Parameters are designed to match the signature of the inheriting classes

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        error_msg = ("GrandCanonicalMonteCarloSampler is not designed to sample!")
        self.logger.error(error_msg)
        raise NotImplementedError(error_msg)


########################################################################################################################
########################################################################################################################
########################################################################################################################

class GCMCSphereSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Base class for carrying out GCMC moves in OpenMM, using a GCMC sphere to sample the system
    """
    def __init__(self, system, topology, temperature, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0,
                 ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None, sphereCentre=None,
                 log='gcmc.log', dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Initialise base
        BaseGrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, ghostFile=ghostFile,
                                                     log=log, dcd=dcd, rst=rst,
                                                     overwrite=overwrite)

        # Initialise variables specific to the GCMC sphere
        self.sphere_radius = sphereRadius
        self.sphere_centre = None
        volume = (4 * np.pi * sphereRadius ** 3) / 3

        if referenceAtoms is not None:
            # Define sphere based on reference atoms
            self.ref_atoms = self.getReferenceAtomIndices(referenceAtoms)
            self.logger.info("GCMC sphere is based on reference atom IDs: {}".format(self.ref_atoms))
        elif sphereCentre is not None:
            # Define sphere based on coordinates
            assert len(sphereCentre) == 3, "Sphere coordinates must be 3D"
            self.sphere_centre = sphereCentre
            self.ref_atoms = None
            self.logger.info("GCMC sphere is fixed in space and centred on {}".format(self.sphere_centre))
        else:
            self.raiseError("A set of atoms or coordinates must be used to define the centre of the sphere!")

        self.logger.info("GCMC sphere radius is {}".format(self.sphere_radius))

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = excessChemicalPotential / self.kT + math.log(volume / standardVolume)
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        self.logger.info("GCMCSphereSampler object initialised")

    def getReferenceAtomIndices(self, ref_atoms):
        """
        Get the index of the atom used to define the centre of the GCMC box

        Parameters
        ----------
        ref_atoms : list
            List of dictionaries containing the atom name, residue name and (optionally) residue ID and chain,
            as marked by keys 'name', 'resname', 'resid' and 'chain'

        Returns
        -------
        atom_indices : list
            Indices of the atoms chosen
        """
        atom_indices = []
        # Convert to list of lists, if not already
        if not all(type(x) == dict for x in ref_atoms):
            self.raiseError("Reference atoms must be a list of dictionaries! {}".format(ref_atoms))

        # Find atom index for each of the atoms used
        for atom_dict in ref_atoms:
            found = False  # Checks if the atom has been found
            # Read in atom data
            name = atom_dict['name']
            resname = atom_dict['resname']
            # Residue ID and chain may not be present
            try:
                resid = atom_dict['resid']
            except:
                resid = None
            try:
                chain = atom_dict['chain']
            except:
                chain = None

            # Loop over all atoms to find one which matches these criteria
            for c, chain_obj in enumerate(self.topology.chains()):
                # Check chain, if specified
                if chain is not None:
                    if c != chain:
                        continue
                for residue in chain_obj.residues():
                    # Check residue name
                    if residue.name != resname:
                        continue
                    # Check residue ID, if specified
                    if resid is not None:
                        if residue.id != resid:
                            continue
                    # Loop over all atoms in this residue to find the one with the right name
                    for atom in residue.atoms():
                        if atom.name == name:
                            atom_indices.append(atom.index)
                            found = True
            if not found:
                self.raiseError("Atom {} of residue {}{} not found!".format(atom_dict['name'],
                                                                            atom_dict['resname'].capitalize(),
                                                                            atom_dict['resid']))

        if len(atom_indices) == 0:
            self.raiseError("No GCMC reference atoms found")

        return atom_indices

    def getSphereCentre(self):
        """
        Update the coordinates of the sphere centre
        Need to make sure it isn't affected by the reference atoms being split across PBCs
        """
        if self.ref_atoms is None:
            self.raiseError("No reference atoms defined, cannot get sphere coordinates...")

        # Calculate the mean coordinate
        self.sphere_centre = np.zeros(3) * unit.nanometers
        for i, atom in enumerate(self.ref_atoms):
            # Need to add on a correction in case the atoms get separated
            correction = np.zeros(3) * unit.nanometers
            if i != 0:
                # Vector from the first reference atom
                vec = self.positions[self.ref_atoms[0]] - self.positions[atom]
                # Correct for PBCs
                for j in range(3):
                    if vec[j] > 0.5 * self.simulation_box[j]:
                        correction[j] = self.simulation_box[j]
                    elif vec[j] < -0.5 * self.simulation_box[j]:
                        correction[j] = -self.simulation_box[j]

            # Add vector and correction onto the running sum
            self.sphere_centre += self.positions[atom] + correction

        # Calculate the average coordinate
        self.sphere_centre /= len(self.ref_atoms)

        return None

    def initialise(self, context, ghostResids=[]):
        """
        Prepare the GCMC sphere for simulation by loading the coordinates from a
        Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        """

        # Load context into sampler
        self.context = context

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Check the symmetry of the box - currently only tolerate cuboidal boxes
        # All off-diagonal box vector components must be zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(box_vectors[i, j]._value, 0.0):
                    self.raiseError("grand only accepts cuboidal simulation cells at this time.")

        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Check size of the GCMC sphere, relative to the box
        for i in range(3):
            if self.sphere_radius > 0.5 * self.simulation_box[i]:
                self.raiseError("GCMC sphere radius cannot be larger than half a box length.")

        # Calculate the centre of the GCMC sphere, if using reference atoms
        if self.ref_atoms is not None:
            self.getSphereCentre()

        # Loop over waters and check which are in/out of the GCMC sphere at the beginning - may be able to replace this with updateGCMCSphere?
        for resid, residue in enumerate(self.topology.residues()):
            if resid not in self.water_resids:
                continue
            for atom in residue.atoms():
                ox_index = atom.index
                break

            vector = self.positions[ox_index] - self.sphere_centre
            # Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]

            # Set the status of this water as appropriate
            if np.linalg.norm(vector) * unit.nanometer <= self.sphere_radius:
                self.setWaterStatus(resid, 1)
            else:
                self.setWaterStatus(resid, 2)

        # Delete ghost waters
        if len(ghostResids) > 0:
            self.deleteGhostWaters(ghostResids)

        return None

    def deleteWatersInGCMCSphere(self):
        """
        Function to delete all of the waters currently present in the GCMC region
        This may be useful the plan is to generate a water distribution for this
        region from scratch. If so, it would be recommended to interleave the GCMC
        sampling with coordinate propagation, as this will converge faster.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the system. Only needs to be supplied if the context
            has changed since the last update

        Returns
        -------
        context : simtk.openmm.Context
            Updated context after deleting the relevant waters
        """
        #  Read in positions of the context and update GCMC box
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        # Loop over all residues to find those of interest
        for resid, residue in enumerate(self.topology.residues()):
            # Make sure this is a water
            if resid not in self.water_resids:
                continue

            # Make sure this is a GCMC water
            if self.getWaterStatusValue(resid) != 1:
                continue

            # Get atom IDs
            atom_ids = []
            for atom in residue.atoms():
                atom_ids.append(atom.index)

            #  Switch off interactions involving the atoms of this residue
            self.adjustSpecificWater(atom_ids, 0.0)

            # Update relevant parameters
            self.setWaterStatus(resid, 0)
            self.N -= 1

        return None

    def updateGCMCSphere(self, state):
        """
        Update the relevant GCMC-sphere related parameters. This also involves monitoring
        which water molecules are in/out of the region

        Parameters
        ----------
        state : simtk.openmm.State
            Current State
        """
        # Make sure the positions are definitely updated
        self.positions = deepcopy(state.getPositions(asNumpy=True))

        # Get the sphere centre, if using reference atoms, otherwise this will be fine
        if self.ref_atoms is not None:
            self.getSphereCentre()

        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Check which waters are in the GCMC region
        for resid, residue in enumerate(self.topology.residues()):
            # Make sure this is a water
            if resid not in self.water_resids:
                continue

            # Get oxygen atom ID
            for atom in residue.atoms():
                ox_index = atom.index
                break

            # Ghost waters automatically count as GCMC waters
            if self.getWaterStatusValue(resid) == 0:
                continue

            # Check if the water is within the sphere
            vector = self.positions[ox_index] - self.sphere_centre
            #  Correct PBCs of this vector - need to make this part cleaner
            for i in range(3):
                if vector[i] >= 0.5 * self.simulation_box[i]:
                    vector[i] -= self.simulation_box[i]
                elif vector[i] <= -0.5 * self.simulation_box[i]:
                    vector[i] += self.simulation_box[i]

            # Set the status of this water as appropriate
            if np.linalg.norm(vector) * unit.nanometer <= self.sphere_radius:
                self.setWaterStatus(resid, 1)
            else:
                self.setWaterStatus(resid, 2)

        # Update lists
        self.N = len(self.getWaterStatusResids(1))

        return None

    def insertRandomWater(self):
        """
        Translate a random ghost to a random point in the GCMC sphere to allow subsequent insertion

        Returns
        -------
        new_positions : simtk.unit.Quantity
            Positions following the 'insertion' of the ghost water
        insert_water : int
            Residue ID of the water to insert
        atom_indices : list
            List of the atom IDs for this molecule
        """
        # Select a ghost water to insert
        ghost_wats = self.getWaterStatusResids(0)
        # Check that there are any ghosts present
        if len(ghost_wats) == 0:
            self.raiseError("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")

        insert_water = np.random.choice(ghost_wats)
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == insert_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Select a point to insert the water (based on O position)
        rand_nums = np.random.randn(3)
        insert_point = self.sphere_centre + (
                self.sphere_radius * np.power(np.random.rand(), 1.0 / 3) * rand_nums) / np.linalg.norm(rand_nums)
        #  Generate a random rotation matrix
        R = random_rotation_matrix()
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atom_indices):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * unit.nanometer
            # Translate to new position
            new_positions[index] = atom_position + insert_point

        return new_positions, insert_water, atom_indices

    def deleteRandomWater(self):
        """
        Choose a random water to be deleted

        Returns
        -------
        delete_water : int
            Resid of the water to delete
        atom_indices : list
            List of the atom IDs for this molecule
        """
        # Cannot carry out deletion if there are no GCMC waters on
        gcmc_wats = self.getWaterStatusResids(1)
        if len(gcmc_wats) == 0:
            return None, None

        # Select a water residue to delete
        delete_water = np.random.choice(gcmc_wats)

        # Get atom indices
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        return delete_water, atom_indices

    def report(self, simulation):
        """
        Function to report any useful data

        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            Simulation object being used
        """
        # Get state
        state = simulation.context.getState(getPositions=True, getVelocities=True)

        # Update GCMC sphere
        self.updateGCMCSphere(state)

        # Calculate rounded acceptance rate and mean N
        if self.n_moves > 0:
            acc_rate = np.round(self.n_accepted * 100.0 / self.n_moves, 4)
        else:
            acc_rate = np.nan
        mean_N = np.round(np.mean(self.Ns), 4)
        # Print out a line describing the acceptance rate and sampling of N
        msg = "{} move(s) completed ({} accepted ({:.4f} %)). Current N = {}. Average N = {:.3f}".format(self.n_moves,
                                                                                                         self.n_accepted,
                                                                                                         acc_rate,
                                                                                                         self.N,
                                                                                                         mean_N)
        print(msg)
        self.logger.info(msg)

        # Write to the file describing which waters are ghosts through the trajectory
        self.writeGhostWaterResids()

        # Append to the DCD and update the restart file
        if self.dcd is not None:
            self.dcd.report(simulation, state)
        if self.restart is not None:
            self.restart.report(simulation, state)

        return None


########################################################################################################################

class StandardGCMCSphereSampler(GCMCSphereSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, ghostFile="gcmc-ghost-wats.txt",
                 referenceAtoms=None, sphereRadius=None, sphereCentre=None, log='gcmc.log', dcd=None, rst=None,
                 overwrite=False):
        """
        Initialise the object to be used for sampling instantaneous water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GCMCSphereSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, ghostFile=ghostFile, referenceAtoms=referenceAtoms,
                                   sphereRadius=sphereRadius, sphereCentre=sphereCentre, log=log, dcd=dcd, rst=rst,
                                   overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.logger.info("StandardGCMCSphereSampler object initialised")

    def move(self, context, n=1):
        """
        Execute a number of GCMC moves on the current system

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)

        # Check change in N
        if len(self.Ns) > 0:
            dN = self.N - self.Ns[-1]
            if abs(dN) > 0:
                self.logger.info('Change in N of {:+} between GCMC batches'.format(dN))

        # Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a random water insertion move on the current system
        """
        # Choose a random site in the sphere to insert a water
        new_positions, resid, atom_indices = self.insertRandomWater()

        # Recouple this water
        self.adjustSpecificWater(atom_indices, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = math.exp(self.B) * math.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(self.positions)
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.setWaterStatus(resid, 1)
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None

    def deletionMove(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Choose a random water in the sphere to be deleted
        resid, atom_indices = self.deleteRandomWater()
        # Deletion may not be possible
        if resid is None:
            return None

        # Switch water off
        self.adjustSpecificWater(atom_indices, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * math.exp(-self.B) * math.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            self.adjustSpecificWater(atom_indices, 1.0)
        else:
            # Update some variables if move is accepted
            self.setWaterStatus(resid, 0)
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


########################################################################################################################

class NonequilibriumGCMCSphereSampler(GCMCSphereSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole, standardVolume=30.345*unit.angstroms**3,
                 adamsShift=0.0, nPertSteps=1, nPropStepsPerPert=1, timeStep=2 * unit.femtoseconds, lambdas=None,
                 ghostFile="gcmc-ghost-wats.txt", referenceAtoms=None, sphereRadius=None, sphereCentre=None,
                 log='gcmc.log', dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling NCMC-enhanced water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        integrator : simtk.openmm.CustomIntegrator
            Integrator to use to propagate the dynamics of the system. Currently want to make sure that this
            is the customised Langevin integrator found in openmmtools which uses BAOAB (VRORV) splitting.
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropStepsPerPert : int
            Number of propagation steps to carry out for
        timeStep : simtk.unit.Quantity
            Time step to use for non-equilibrium integration during the propagation steps
        lambdas : list
            Series of lambda values corresponding to the pathway over which the molecules are perturbed
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        referenceAtoms : list
            List containing dictionaries describing the atoms to use as the centre of the GCMC region
            Must contain 'name' and 'resname' as keys, and optionally 'resid' (recommended) and 'chain'
            e.g. [{'name': 'C1', 'resname': 'LIG', 'resid': '123'}]
        sphereRadius : simtk.unit.Quantity
            Radius of the spherical GCMC region
        sphereCentre : simtk.unit.Quantity
            Coordinates around which the GCMC sphere is based
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class
        GCMCSphereSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, ghostFile=ghostFile, referenceAtoms=referenceAtoms,
                                   sphereRadius=sphereRadius, sphereCentre=sphereCentre, log=log, dcd=dcd, rst=rst,
                                   overwrite=overwrite)

        self.velocities = None  # Need to store velocities for this type of sampling

        # Load in extra NCMC variables
        if lambdas is not None:
            # Read in set of lambda values, if specified
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            # Otherwise, assume they are evenly distributed
            self.n_pert_steps = nPertSteps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        self.n_pert_steps = nPertSteps
        self.n_prop_steps_per_pert = nPropStepsPerPert
        self.time_step = timeStep.in_units_of(unit.picosecond)
        self.protocol_time = (self.n_pert_steps + 1) * self.n_prop_steps_per_pert * self.time_step
        self.logger.info("Each NCMC move will be executed over a total of {}".format(self.protocol_time))

        self.insert_works = []  # Store work values of moves
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0  # Number of moves rejected because the water left the sphere

        # Define a compound integrator
        self.compound_integrator = openmm.CompoundIntegrator()
        # Add the MD integrator
        self.compound_integrator.addIntegrator(integrator)
        # Create and add the nonequilibrium integrator
        self.ncmc_integrator = NonequilibriumLangevinIntegrator(temperature=temperature,
                                                                collision_rate=1.0/unit.picosecond,
                                                                timestep=self.time_step, splitting="V R O R V")
        self.compound_integrator.addIntegrator(self.ncmc_integrator)
        # Set the compound integrator to the MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        self.logger.info("NonequilibriumGCMCSphereSampler object initialised")

    def move(self, context, n=1):
        """
        Carry out a nonequilibrium GCMC move

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Update GCMC region based on current state
        self.updateGCMCSphere(state)

        # Set to NCMC integrator
        self.compound_integrator.setCurrentIntegrator(1)

        #  Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        # Set to MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        return None

    def insertionMove(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        # Choose a random site in the sphere to insert a water
        new_positions, resid, atom_indices = self.insertRandomWater()

        # Need to update the context positions
        self.context.setPositions(new_positions)

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, self.lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Store the protocol work
        self.insert_works.append(protocol_work)

        # Update variables and GCMC sphere
        self.setWaterStatus(resid, 1)
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        self.updateGCMCSphere(state)

        # Check which waters are in the sphere
        wats_in_sphere = self.getWaterStatusResids(1)

        # Calculate acceptance probability
        if resid not in wats_in_sphere:
            # If the inserted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = -1
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to water leaving the GCMC sphere")
        elif explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = math.exp(self.B) * math.exp(-protocol_work/self.kT) / self.N  # Here N is the new value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.setWaterStatus(resid, 0)
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            self.N = len(wats_in_sphere)
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

    def deletionMove(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Store initial positions
        old_positions = deepcopy(self.positions)

        # Choose a random water in the sphere to be deleted
        resid, atom_indices = self.deleteRandomWater()
        # Deletion may not be possible
        if resid is None:
            return None

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, self.lambdas[-(2+i)])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        self.delete_works.append(protocol_work)

        # Update variables and GCMC sphere
        # Leaving the water as 'on' here to check that the deleted water doesn't leave
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = state.getPositions(asNumpy=True)
        old_N = self.N
        self.updateGCMCSphere(state)

        # Check which waters are in the sphere
        wats_in_sphere = self.getWaterStatusResids(1)

        # Calculate acceptance probability
        if resid not in wats_in_sphere:
            # If the deleted water leaves the sphere, the move cannot be reversed and therefore cannot be accepted
            acc_prob = 0
            self.n_left_sphere += 1
            self.logger.info("Move rejected due to water leaving the GCMC sphere")
        elif explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = old_N * math.exp(-self.B) * math.exp(-protocol_work/self.kT)  # N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 1.0)
            self.context.setPositions(old_positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(old_positions)
            self.velocities = -self.velocities
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
            self.updateGCMCSphere(state)
        else:
            # Update some variables if move is accepted
            self.setWaterStatus(resid, 0)
            self.N = len(wats_in_sphere) - 1  # Accounting for the deleted water
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.updateGCMCSphere(state)

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []

        # NCMC-specific variables
        self.insert_works = []
        self.delete_works = []
        self.n_explosions = 0
        self.n_left_sphere = 0

        return None


########################################################################################################################
########################################################################################################################
########################################################################################################################

class GCMCSystemSampler(BaseGrandCanonicalMonteCarloSampler):
    """
    Base class for carrying out GCMC moves in OpenMM, sampling the whole system with GCMC
    """
    def __init__(self, system, topology, temperature, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Overwrite any data already present
        """
        # Initialise base
        BaseGrandCanonicalMonteCarloSampler.__init__(self, system, topology, temperature, ghostFile=ghostFile, log=log,
                                                     dcd=dcd, rst=rst, overwrite=overwrite)

        # Read in simulation box lengths
        self.simulation_box = np.array([boxVectors[0, 0]._value,
                                        boxVectors[1, 1]._value,
                                        boxVectors[2, 2]._value]) * unit.nanometer
        volume = self.simulation_box[0] * self.simulation_box[1] * self.simulation_box[2]

        # Set or calculate the Adams value for the simulation
        if adams is not None:
            self.B = adams
        else:
            # Calculate Bequil from the chemical potential and volume
            self.B = excessChemicalPotential / self.kT + math.log(volume / standardVolume)
            # Shift B from Bequil if necessary
            self.B += adamsShift

        self.logger.info("Simulating at an Adams (B) value of {}".format(self.B))

        self.logger.info("GCMCSystemSampler object initialised")

    def initialise(self, context, ghostResids):
        """
        Prepare the GCMC sphere for simulation by loading the coordinates from a
        Context object.

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        ghostResids : list
            List of residue IDs corresponding to the ghost waters added
        """
        # Load context into sampler
        self.context = context

        # Load in positions and box vectors from context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        # Check the symmetry of the box - currently only tolerate cuboidal boxes
        # All off-diagonal box vector components must be zero
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if not np.isclose(box_vectors[i, j]._value, 0.0):
                    self.raiseError("grand only accepts cuboidal simulation cells at this time.")

        self.simulation_box = np.array([box_vectors[0, 0]._value,
                                        box_vectors[1, 1]._value,
                                        box_vectors[2, 2]._value]) * unit.nanometer

        # Delete ghost waters
        self.deleteGhostWaters(ghostResids)

        # Count N
        self.N = np.sum(self.water_status)

        return None

    def insertRandomWater(self):
        """
        Translate a random ghost to a random point in the simulation box to allow subsequent insertion

        Returns
        -------
        new_positions : simtk.unit.Quantity
            Positions following the 'insertion' of the ghost water
        insert_water : int
            Residue ID of the water to insert
        atom_indices : list
            List of the atom IDs for this molecule
        """
        # Select a ghost water to insert
        ghost_wats = self.getWaterStatusResids(0)
        # Check that there are any ghosts present
        if len(ghost_wats) == 0:
            self.raiseError("No ghost water molecules left, so insertion moves cannot occur - add more ghost waters")

        insert_water = np.random.choice(ghost_wats)
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == insert_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        # Select a point to insert the water (based on O position)
        insert_point = np.random.rand(3) * self.simulation_box
        #  Generate a random rotation matrix
        R = random_rotation_matrix()
        new_positions = deepcopy(self.positions)
        for i, index in enumerate(atom_indices):
            #  Translate coordinates to an origin defined by the oxygen atom, and normalise
            atom_position = self.positions[index] - self.positions[atom_indices[0]]
            # Rotate about the oxygen position
            if i != 0:
                vec_length = np.linalg.norm(atom_position)
                atom_position = atom_position / vec_length
                # Rotate coordinates & restore length
                atom_position = vec_length * np.dot(R, atom_position) * unit.nanometer
            # Translate to new position
            new_positions[index] = atom_position + insert_point

        return new_positions, insert_water, atom_indices

    def deleteRandomWater(self):
        """
        Choose a random water to be deleted

        Returns
        -------
        delete_water : int
            Resid of the water to delete
        atom_indices : list
            List of the atom IDs for this molecule
        """
        # Cannot carry out deletion if there are no GCMC waters on
        gcmc_wats = self.getWaterStatusResids(1)
        if len(gcmc_wats) == 0:
            return None, None

        # Select a water residue to delete
        delete_water = np.random.choice(gcmc_wats)
        atom_indices = []
        for resid, residue in enumerate(self.topology.residues()):
            if resid == delete_water:
                for atom in residue.atoms():
                    atom_indices.append(atom.index)

        return delete_water, atom_indices


########################################################################################################################

class StandardGCMCSystemSampler(GCMCSystemSampler):
    """
    Class to carry out instantaneous GCMC moves in OpenMM
    """
    def __init__(self, system, topology, temperature, adams=None, excessChemicalPotential=-6.09*unit.kilocalories_per_mole,
                 standardVolume=30.345*unit.angstroms**3, adamsShift=0.0, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', dcd=None, rst=None, overwrite=False):
        """
        Initialise the object to be used for sampling instantaneous water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class - don't need any more initialisation for the instantaneous sampler
        GCMCSystemSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, boxVectors=boxVectors, ghostFile=ghostFile, log=log,
                                   dcd=dcd, rst=rst, overwrite=overwrite)

        self.energy = None  # Need to save energy
        self.logger.info("StandardGCMCSystemSampler object initialised")

    def move(self, context, n=1):
        """
        Execute a number of GCMC moves on the current system

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getEnergy=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.energy = state.getPotentialEnergy()

        # Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        return None

    def insertionMove(self):
        """
        Carry out a random water insertion move on the current system
        """
        # Insert a ghost water to a random site
        new_positions, resid, atom_indices = self.insertRandomWater()

        # Recouple this water
        self.adjustSpecificWater(atom_indices, 1.0)

        self.context.setPositions(new_positions)
        # Calculate new system energy and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = math.exp(self.B) * math.exp(-(final_energy - self.energy) / self.kT) / (self.N + 1)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            # Switch off nonbonded interactions involving this water
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(self.positions)  # Not sure this is necessary...
        else:
            # Update some variables if move is accepted
            self.positions = deepcopy(new_positions)
            self.setWaterStatus(resid, 1)
            self.N += 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None

    def deletionMove(self):
        """
        Carry out a random water deletion move on the current system
        """
        # Choose a random water to be deleted
        resid, atom_indices = self.deleteRandomWater()
        # Deletion may not be possible
        if resid is None:
            return None

        # Switch water off
        self.adjustSpecificWater(atom_indices, 0.0)
        # Calculate energy of new state and acceptance probability
        final_energy = self.context.getState(getEnergy=True).getPotentialEnergy()
        acc_prob = self.N * math.exp(-self.B) * math.exp(-(final_energy - self.energy) / self.kT)
        self.acceptance_probabilities.append(acc_prob)

        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Switch the water back on if the move is rejected
            self.adjustSpecificWater(atom_indices, 1.0)
        else:
            # Update some variables if move is accepted
            self.setWaterStatus(resid, 0)
            self.N -= 1
            self.n_accepted += 1
            # Update energy
            self.energy = final_energy

        return None


########################################################################################################################

class NonequilibriumGCMCSystemSampler(GCMCSystemSampler):
    """
    Class to carry out GCMC moves in OpenMM, using nonequilibrium candidate Monte Carlo (NCMC)
    to boost acceptance rates
    """
    def __init__(self, system, topology, temperature, integrator, adams=None,
                 excessChemicalPotential=-6.09*unit.kilocalories_per_mole, standardVolume=30.345*unit.angstroms**3,
                 adamsShift=0.0, nPertSteps=1, nPropStepsPerPert=1, timeStep=2 * unit.femtoseconds, boxVectors=None,
                 ghostFile="gcmc-ghost-wats.txt", log='gcmc.log', dcd=None, rst=None, overwrite=False,
                 lambdas=None):
        """
        Initialise the object to be used for sampling NCMC-enhanced water insertion/deletion moves

        Parameters
        ----------
        system : simtk.openmm.System
            System object to be used for the simulation
        topology : simtk.openmm.app.Topology
            Topology object for the system to be simulated
        temperature : simtk.unit.Quantity
            Temperature of the simulation, must be in appropriate units
        integrator : simtk.openmm.CustomIntegrator
            Integrator to use to propagate the dynamics of the system. Currently want to make sure that this
            is the customised Langevin integrator found in openmmtools which uses BAOAB (VRORV) splitting.
        adams : float
            Adams B value for the simulation (dimensionless). Default is None,
            if None, the B value is calculated from the box volume and chemical
            potential
        excessChemicalPotential : simtk.unit.Quantity
            Excess chemical potential of the system that the simulation should be in equilibrium with, default is
            -6.09 kcal/mol. This should be the hydration free energy of water, and may need to be changed for specific
            simulation parameters.
        standardVolume : simtk.unit.Quantity
            Standard volume of water - corresponds to the volume per water molecule in bulk. The default value is 30.345 A^3
        adamsShift : float
            Shift the B value from Bequil, if B isn't explicitly set. Default is 0.0
        nPertSteps : int
            Number of pertubation steps over which to shift lambda between 0 and 1 (or vice versa).
        nPropStepsPerPert : int
            Number of propagation steps to carry out for
        timeStep : simtk.unit.Quantity
            Time step to use for non-equilibrium integration during the propagation steps
        lambdas : list
            Series of lambda values corresponding to the pathway over which the molecules are perturbed
        boxVectors : simtk.unit.Quantity
            Box vectors for the simulation cell
        ghostFile : str
            Name of a file to write out the residue IDs of ghost water molecules. This is
            useful if you want to visualise the sampling, as you can then remove these waters
            from view, as they are non-interacting. Default is 'gcmc-ghost-wats.txt'
        log : str
            Name of the log file to write out
        dcd : str
            Name of the DCD file to write the system out to
        rst : str
            Name of the restart file to write out (.pdb or .rst7)
        overwrite : bool
            Indicates whether to overwrite already existing data
        """
        # Initialise base class
        GCMCSystemSampler.__init__(self, system, topology, temperature, adams=adams,
                                   excessChemicalPotential=excessChemicalPotential, standardVolume=standardVolume,
                                   adamsShift=adamsShift, boxVectors=boxVectors, ghostFile=ghostFile, log=log, dcd=dcd,
                                   rst=rst, overwrite=overwrite)

        # Load in extra NCMC variables
        if lambdas is not None:
            # Read in set of lambda values, if specified
            assert np.isclose(lambdas[0], 0.0) and np.isclose(lambdas[-1], 1.0), "Lambda series must start at 0 and end at 1"
            self.lambdas = lambdas
            self.n_pert_steps = len(self.lambdas) - 1
        else:
            # Otherwise, assume they are evenly distributed
            self.n_pert_steps = nPertSteps
            self.lambdas = np.linspace(0.0, 1.0, self.n_pert_steps + 1)

        self.n_prop_steps_per_pert = nPropStepsPerPert
        self.time_step = timeStep.in_units_of(unit.picosecond)
        self.protocol_time = (self.n_pert_steps + 1) * self.n_prop_steps_per_pert * self.time_step
        self.logger.info("Each NCMC move will be executed over a total of {}".format(self.protocol_time))

        self.velocities = None  # Need to store velocities for this type of sampling

        self.insert_works = []  # Store work values of moves
        self.delete_works = []
        self.n_explosions = 0

        # Define a compound integrator
        self.compound_integrator = openmm.CompoundIntegrator()
        # Add the MD integrator
        self.compound_integrator.addIntegrator(integrator)
        # Create and add the nonequilibrium integrator
        self.ncmc_integrator = NonequilibriumLangevinIntegrator(temperature=temperature,
                                                                collision_rate=1.0/unit.picosecond,
                                                                timestep=self.time_step, splitting="V R O R V")
        self.compound_integrator.addIntegrator(self.ncmc_integrator)
        # Set the compound integrator to the MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        self.logger.info("NonequilibriumGCMCSystemSampler object initialised")

    def move(self, context, n=1):
        """
        Carry out a nonequilibrium GCMC move

        Parameters
        ----------
        context : simtk.openmm.Context
            Current context of the simulation
        n : int
            Number of moves to execute
        """
        # Read in positions
        self.context = context
        state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
        self.positions = deepcopy(state.getPositions(asNumpy=True))
        self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        # Set to NCMC integrator
        self.compound_integrator.setCurrentIntegrator(1)

        #  Execute moves
        for i in range(n):
            # Insert or delete a water, based on random choice
            if np.random.randint(2) == 1:
                # Attempt to insert a water
                self.insertionMove()
            else:
                # Attempt to delete a water
                self.deletionMove()
            self.n_moves += 1
            self.Ns.append(self.N)

        # Set to MD integrator
        self.compound_integrator.setCurrentIntegrator(0)

        return None

    def insertionMove(self):
        """
        Carry out a nonequilibrium insertion move for a random water molecule
        """
        # Insert a ghost water to a random site
        new_positions, resid, atom_indices = self.insertRandomWater()

        # Need to update the context positions
        self.context.setPositions(new_positions)

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, self.lambdas[i+1])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        self.insert_works.append(protocol_work)

        if explosion:
            acc_prob = -1
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = math.exp(self.B) * math.exp(-protocol_work/self.kT) / (self.N + 1)  # Here N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 0.0)
            self.context.setPositions(self.positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(self.positions)
            self.velocities = -self.velocities
        else:
            # Update some variables if move is accepted
            self.N += 1
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))
            self.setWaterStatus(resid, 1)

        return None

    def deletionMove(self):
        """
        Carry out a nonequilibrium deletion move for a random water molecule
        """
        # Choose a random water to be deleted
        resid, atom_indices = self.deleteRandomWater()
        # Deletion may not be possible
        if resid is None:
            return None

        # Start running perturbation and propagation kernels
        protocol_work = 0.0 * unit.kilocalories_per_mole
        explosion = False
        self.ncmc_integrator.step(self.n_prop_steps_per_pert)
        for i in range(self.n_pert_steps):
            state = self.context.getState(getEnergy=True)
            energy_initial = state.getPotentialEnergy()
            # Adjust interactions of this water
            self.adjustSpecificWater(atom_indices, self.lambdas[-(2+i)])
            state = self.context.getState(getEnergy=True)
            energy_final = state.getPotentialEnergy()
            protocol_work += energy_final - energy_initial
            # Propagate the system
            try:
                self.ncmc_integrator.step(self.n_prop_steps_per_pert)
            except:
                print("Caught explosion!")
                explosion = True
                self.n_explosions += 1
                break

        # Get the protocol work
        self.delete_works.append(protocol_work)

        if explosion:
            acc_prob = 0
            self.logger.info("Move rejected due to an instability during integration")
        else:
            # Calculate acceptance probability based on protocol work
            acc_prob = self.N * math.exp(-self.B) * math.exp(-protocol_work/self.kT)  # N is the old value

        self.acceptance_probabilities.append(acc_prob)

        # Update or reset the system, depending on whether the move is accepted or rejected
        if acc_prob < np.random.rand() or np.isnan(acc_prob):
            # Need to revert the changes made if the move is to be rejected
            self.adjustSpecificWater(atom_indices, 1.0)
            self.context.setPositions(self.positions)
            self.context.setVelocities(-self.velocities)  # Reverse velocities on rejection
            self.positions = deepcopy(self.positions)
            self.velocities = -self.velocities
        else:
            # Update some variables if move is accepted
            self.setWaterStatus(resid, 0)
            self.N -= 1
            self.n_accepted += 1
            state = self.context.getState(getPositions=True, enforcePeriodicBox=True, getVelocities=True)
            self.positions = deepcopy(state.getPositions(asNumpy=True))
            self.velocities = deepcopy(state.getVelocities(asNumpy=True))

        return None

    def reset(self):
        """
        Reset counted values (such as number of total or accepted moves) to zero
        """
        self.logger.info('Resetting any tracked variables...')
        self.n_accepted = 0
        self.n_moves = 0
        self.Ns = []
        self.acceptance_probabilities = []

        # NCMC-specific variables
        self.insert_works = []
        self.delete_works = []
        self.n_explosions = 0

        return None

