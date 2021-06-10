# -*- coding: utf-8 -*-

"""
Description
-----------
Set of functions to calibrate the excess chemical potential and standard state volume of water

Marley Samways
"""

import numpy as np
import pymbar
import openmmtools
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import grand

def get_lambda_values(lambda_in):
    """
    Calculate the lambda_sterics and lambda_electrostatics values for a given lambda.
    Electrostatics are decoupled from lambda=1 to 0.5, and sterics are decoupled from
    lambda=0.5 to 0.

    Parameters
    ----------
    lambda_in : float
        Input lambda value

    Returns
    -------
    lambda_vdw : float
        Lambda value for steric interactions
    lambda_ele : float
        Lambda value for electrostatic interactions
    """
    if lambda_in > 1.0:
        # Set both values to 1.0 if lambda > 1
        lambda_vdw = 1.0
        lambda_ele = 1.0
    elif lambda_in < 0.0:
        # Set both values to 0.0 if lambda < 0
        lambda_vdw = 0.0
        lambda_ele = 0.0
    else:
        # Scale values between 0 and 1
        lambda_vdw = min([1.0, 2.0*lambda_in])
        lambda_ele = max([0.0, 2.0*(lambda_in-0.5)])
    return lambda_vdw, lambda_ele


def calc_mu_ex(system, topology, positions, box_vectors, temperature, n_lambdas, n_samples, n_equil, log_file):
    """
    Calculate the excess chemical potential of a water molecule in a given system,
    as the hydration free energy, using MBAR

    Parameters
    ----------
    system : simtk.openmm.System
        System of interest
    topology : simtk.openmm.app.Topology
        Topology of the system
    positions : simtk.unit.Quantity
        Initial positions for the simulation
    box_vectors : simtk.unit.Quantity
        Periodic box vectors for the system
    temperature : simtk.unit.Quantity
        Temperature of the simulation
    n_lambdas : int
        Number of lambda values
    n_samples : int
        Number of energy samples to collect at each lambda value
    n_equil : int
        Number of MD steps to run between each sample
    log_file : str
        Name of the log file to write out

    Returns
    -------
    dG : simtk.unit.Quantity
        Calculated free energy value
    """
    # Use the BAOAB integrator to sample the equilibrium distribution
    integrator = openmmtools.integrators.BAOABIntegrator(temperature, 1.0/picosecond, 0.002*picoseconds)

    # Name the log file, if not already done
    if log_file is None:
        'dG-{}l-{}sa-{}st.log'.format(n_lambdas, n_samples, n_equil)

    # Define a GCMC sampler object, just to allow easy switching of a water - won't use this to sample
    gcmc_mover = grand.samplers.BaseGrandCanonicalMonteCarloSampler(system=system, topology=topology,
                                                                    temperature=temperature,
                                                                    log=log_file,
                                                                    ghostFile='calc_mu-ghosts.txt',
                                                                    overwrite=True)
    # Remove unneeded ghost file
    os.remove('calc_mu-ghosts.txt')

    # Testing with barostat
    pressure = 1 * bar
    system.addForce(MonteCarloBarostat(pressure, temperature, 25))

    # IDs of the atoms to switch on/off
    wat_ids = []
    for residue in topology.residues():
        for atom in residue.atoms():
            wat_ids.append(atom.index)
        break  # Make sure to stop after the first water

    # Define the platform, first try CUDA, then OpenCL, then CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            #platform.setPropertyDefaultValue('Precision', 'mixed')
        except:
            platform = Platform.getPlatformByName('CPU')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.context.setPeriodicBoxVectors(*box_vectors)

    # Make sure the GCMC sampler has access to the Context
    gcmc_mover.context = simulation.context

    lambdas = np.linspace(1.0, 0.0, n_lambdas)  # Lambda values to use
    U = np.zeros((n_lambdas, n_lambdas, n_samples))  # Energy values calculated

    # Simulate the system at each lambda window
    for i in range(n_lambdas):
        # Set lambda values
        gcmc_mover.logger.info('Simulating at lambda = {:.4f}'.format(np.round(lambdas[i], 4)))
        gcmc_mover.adjustSpecificWater(wat_ids, lambdas[i])
        for k in range(n_samples):
            # Run production MD
            simulation.step(n_equil)
            box_vectors = simulation.context.getState(getPositions=True).getPeriodicBoxVectors()
            volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]
            # Calculate energy at each lambda value
            for j in range(n_lambdas):
                # Set lambda value
                gcmc_mover.adjustSpecificWater(wat_ids, lambdas[j])
                # Calculate energy
                #U[i, j, k] = simulation.context.getState(getEnergy=True).getPotentialEnergy() / gcmc_mover.kT
                # Calculate energy (with volume correction)
                U[i, j, k] = (simulation.context.getState(getEnergy=True).getPotentialEnergy() + (pressure * volume * AVOGADRO_CONSTANT_NA) ) / gcmc_mover.kT
            # Reset lambda value
            gcmc_mover.adjustSpecificWater(wat_ids, lambdas[i])

    # Calculate equilibration & number of uncorrelated samples
    N_k = np.zeros(n_lambdas, np.int32)
    for i in range(n_lambdas):
        n_equil, g, neff_max = pymbar.timeseries.detectEquilibration(U[i, i, :])
        indices = pymbar.timeseries.subsampleCorrelatedData(U[i, i, :], g=g)
        N_k[i] = len(indices)
        U[i, :, 0:N_k[i]] = U[i, :, indices].T

    # Calculate free energy differences
    mbar = pymbar.MBAR(U, N_k)
    results = mbar.getFreeEnergyDifferences()
    deltaG_ij = results[0]
    ddeltaG_ij = results[1]

    # Extract overall free energy change
    dG = -deltaG_ij[0, -1]

    # Write out intermediate free energies
    for i in range(n_lambdas):
        dG_i = (-deltaG_ij[0, i] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
        gcmc_mover.logger.info('Free energy ({:.3f} -> {:.3f}) = {}'.format(lambdas[0], lambdas[i], dG_i))

    # Convert free energy to kcal/mol
    dG = (dG * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)
    dG_err = (ddeltaG_ij[0, -1] * gcmc_mover.kT).in_units_of(kilocalorie_per_mole)

    gcmc_mover.logger.info('Excess chemical potential = {}'.format(dG))
    gcmc_mover.logger.info('Estimated error = {}'.format(dG_err))

    return dG


def calc_std_volume(system, topology, positions, box_vectors, temperature, n_samples, n_equil):
    """
    Calculate the standard volume of a given system and parameters, this is the effective volume
    of a single molecule

    Parameters
    ----------
    system : simtk.openmm.System
        System of interest
    topology : simtk.openmm.app.Topology
        Topology of the system
    positions : simtk.unit.Quantity
        Initial positions for the simulation
    box_vectors : simtk.unit.Quantity
        Periodic box vectors for the system
    temperature : simtk.unit.Quantity
        Temperature of the simulation
    n_samples : int
        Number of volume samples to collect
    n_equil : int
        Number of MD steps to run between each sample

    Returns
    -------
    std_volume : simtk.unit.Quantity
        Calculated free energy value
    """
    # Use the BAOAB integrator to sample the equilibrium distribution
    integrator = openmmtools.integrators.BAOABIntegrator(temperature, 1.0 / picosecond, 0.002 * picoseconds)

    # Define the platform, first try CUDA, then OpenCL, then CPU
    try:
        platform = Platform.getPlatformByName('CUDA')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    except:
        try:
            platform = Platform.getPlatformByName('OpenCL')
            #platform.setPropertyDefaultValue('Precision', 'mixed')
        except:
            platform = Platform.getPlatformByName('CPU')

    # Create a simulation object
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.context.setPeriodicBoxVectors(*box_vectors)

    # Count number of residues
    n_molecules = 0
    for residue in topology.residues():
        n_molecules += 1

    # Collect volume samples
    volume_list = []
    for i in range(n_samples):
        # Run a short amount of MD
        simulation.step(n_equil)
        # Calculate volume & then volume per molecule
        state = simulation.context.getState(getPositions=True)
        box_vectors = state.getPeriodicBoxVectors(asNumpy=True)
        volume = box_vectors[0, 0] * box_vectors[1, 1] * box_vectors[2, 2]
        volume_list.append(volume / n_molecules)

    # Calculate mean volume per molecule
    std_volume = sum(volume_list) / len(volume_list)

    return std_volume

