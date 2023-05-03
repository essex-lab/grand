# -*- coding: utf-8 -*-

"""
Description
-----------
Functions to provide support for grand canonical sampling in OpenMM.
These functions are not used during the simulation, but will be relevant in setting up
simulations and processing results

Marley Samways
Ollie Melling
"""

import os
import numpy as np
import mdtraj
import parmed
from simtk import unit
from simtk.openmm import app
from copy import deepcopy
from scipy.cluster import hierarchy
import warnings


class PDBRestartReporter(object):
    """
    *Very* basic class to write PDB files as a basic form of restarter
    """
    def __init__(self, filename, topology):
        """
        Load in the name of the file and the topology of the system

        Parameters
        ----------
        filename : str
            Name of the PDB file to write out
        topology : simtk.openmm.app.Topology
            Topology object for the system of interest
        """
        self.filename = filename
        self.topology = topology

    def report(self, simulation, state):
        """
        Write out a PDB of the current state

        Parameters
        ----------
        simulation : simtk.openmm.app.Simulation
            Simulation object being used
        state : simtk.openmm.State
            Current State of the simulation
        """
        # Read the positions from the state
        positions = state.getPositions()
        # Write the PDB out
        with open(self.filename, 'w') as f:
            app.PDBFile.writeFile(topology=self.topology, positions=positions, file=f)

        return None


def get_data_file(filename):
    """
    Get the absolute path of one of the data files included in the package

    Parameters
    ----------
    filename : str
        Name of the file

    Returns
    -------
    filepath : str
        Name of the file including the path
    """
    filepath = os.path.join(os.path.dirname(__file__), "data", filename)
    if os.path.isfile(filepath):
        return filepath
    else:
        raise Exception("{} does not exist!".format(filepath))


def add_ghosts(topology, positions, ff='tip3p', n=10, pdb='gcmc-extra-wats.pdb'):
    """
    Function to add water molecules to a topology, as extras for GCMC
    This is to avoid changing the number of particles throughout a simulation
    Instead, we can just switch between 'ghost' and 'real' waters...

    Notes
    -----
    Ghosts currently all added to a new chain
    Residue numbering continues from the existing PDB numbering

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        Topology of the initial system
    positions : simtk.unit.Quantity
        Atomic coordinates of the initial system
    ff : str
        Water forcefield to use. Currently only TIP3P is supported. 
    n : int
        Number of waters to add to the system
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : simtk.openmm.app.Topology
        Topology of the system after modification
    modeller.positions : simtk.unit.Quantity
        Atomic positions of the system after modification
    ghosts : list
        List of the residue numbers (counting from 0) of the ghost
        waters added to the system.
    """
    # Create a Modeller instance of the system
    modeller = app.Modeller(topology=topology, positions=positions)

    # Read the chain IDs
    chain_ids = []
    for chain in modeller.topology.chains():
        chain_ids.append(chain.id)

    # Read in simulation box size
    box_vectors = topology.getPeriodicBoxVectors()
    box_size = np.array([box_vectors[0][0]._value,
                         box_vectors[1][1]._value,
                         box_vectors[2][2]._value]) * unit.nanometer

    # Load topology of water model
    assert ff.lower() == "tip3p", "Water model must be TIP3P!"
    water = app.PDBFile(file=get_data_file("{}.pdb".format(ff.lower())))

    # Add multiple copies of the same water, then write out a pdb (for visualisation)
    ghosts = []
    for i in range(n):
        # Read in template water positions
        positions = water.positions

        # Need to translate the water to a random point in the simulation box
        new_centre = np.random.rand(3) * box_size
        new_positions = deepcopy(water.positions)
        for i in range(len(positions)):
            new_positions[i] = positions[i] + new_centre - positions[0]

        # Add the water to the model and include the resid in a list
        modeller.add(addTopology=water.topology, addPositions=new_positions)
        ghosts.append(modeller.topology._numResidues - 1)

    # Take the ghost chain as the one after the last chain (alphabetically)
    new_chain = chr(((ord(chain_ids[-1]) - 64) % 26) + 65)

    # Renumber all ghost waters and assign them to the new chain
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            residue.id = str(((resid - 1) % 9999) + 1)
            residue.chain.id = new_chain

    # Write the new topology and positions to a PDB file
    if pdb is not None:
        with open(pdb, 'w') as f:
            app.PDBFile.writeFile(topology=modeller.topology, positions=modeller.positions, file=f, keepIds=True)

    return modeller.topology, modeller.positions, ghosts


def remove_ghosts(topology, positions, ghosts=None, pdb='gcmc-removed-ghosts.pdb'):
    """
    Function to remove ghost water molecules from a topology, after a simulation.
    This is so that a structure can then be used to run further analysis without ghost
    waters disturbing the system.

    Parameters
    ----------
    topology : simtk.openmm.app.Topology
        Topology of the initial system
    positions : simtk.unit.Quantity
        Atomic coordinates of the initial system
    ghosts : list
        List of residue IDs for the ghost waters to be deleted
    pdb : str
        Name of the PDB file to write containing the updated system
        This will be useful for visualising the results obtained.

    Returns
    -------
    modeller.topology : simtk.openmm.app.Topology
        Topology of the system after modification
    modeller.positions : simtk.unit.Quantity
        Atomic positions of the system after modification
    """
    # Do nothing if no ghost waters are specified
    if ghosts is None:
        raise Exception("No ghost waters defined! Nothing to do.")

    # Create a Modeller instance
    modeller = app.Modeller(topology=topology, positions=positions)

    # Find the residues which need to be removed, and delete them
    delete_waters = []  # Residue objects for waters to be deleted
    for resid, residue in enumerate(modeller.topology.residues()):
        if resid in ghosts:
            delete_waters.append(residue)
    modeller.delete(toDelete=delete_waters)

    # Save PDB file
    if pdb is not None:
        with open(pdb, 'w') as f:
            app.PDBFile.writeFile(topology=modeller.topology, positions=modeller.positions, file=f)

    return modeller.topology, modeller.positions


def read_ghosts_from_file(ghost_file):
    """
    Read in the ghost water residue IDs from each from in the ghost file

    Parameters
    ----------
    ghost_file : str
        File containing the IDs of the ghost residues in each frame

    Returns
    -------
    ghost_resids : list
        List of lists, containing residue IDs for each frame
    """
    # Read in residue IDs for the ghost waters in each frame
    ghost_resids = []
    with open(ghost_file, 'r') as f:
        for line in f.readlines():
            ghost_resids.append([int(resid) for resid in line.split(",")])

    return ghost_resids


def read_prepi(filename):
    """
    Function to read in some atomic data and bonding information from an AMBER prepi file

    Parameters
    ----------
    filename : str
        Name of the prepi file

    Returns
    -------
    atom_data : list
        A list containing a list for each atom, of the form [name, type, charge], where each are strings
    bonds : list
        A list containing one list per bond, of the form [name1, name2]
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    atom_dict = {}  #  Indicates the ID number of each atom name
    atom_data = []  #  Will store the name, type and charge of each atom
    bonds = []  #  List of bonds between atoms
    for i, line_i in enumerate(lines):
        line_data = line_i.split()
        # First read in the data from the atom lines
        if len(line_data) > 10:
            atom_id = line_data[0]
            atom_name = line_data[1]
            atom_type = line_data[2]
            bond_id = line_data[4]
            atom_charge = line_data[10]

            # Ignore dummies
            if atom_type == 'DU':
                continue

            atom_dict[atom_id] = atom_name
            atom_data.append([atom_name, atom_type, atom_charge])
            # Double checking the atom isn't bonded to a dummy before writing
            if int(bond_id) > 3:
                bond_name = atom_dict[bond_id]
                bonds.append([atom_name, bond_name])
        # Now read in the data from the loop-completion lines
        elif line_i.startswith('LOOP'):
            for line_j in lines[i + 1:]:
                if len(line_j.split()) == 2:
                    bonds.append(line_j.split())
                else:
                    break

    return atom_data, bonds


def write_conect(pdb, resname, prepi, output):
    """
    Take in a PDB and write out a new one, including CONECT lines for a specified residue, given a .prepi file
    Should make it easy to run this on more residues at a time - though for now it can just be run separately per residue
    but this isn't ideal...

    Parameters
    ----------
    pdb : str
        Name of the input PDB file
    resname : str
        Name of the residue of interest
    prepi : str
        Name of the prepi file
    output : str
        Name of the output PDB file
    """
    # Read in bonds from prepi file
    _, bond_list = read_prepi(prepi)

    resids_done = []  # List of completed residues

    conect_lines = []  # List of CONECT lines to add

    with open(pdb, 'r') as f:
        pdb_lines = f.readlines()

    for i, line_i in enumerate(pdb_lines):
        if not any([line_i.startswith(x) for x in ['ATOM', 'HETATM']]):
            continue

        if line_i[17:21].strip() == resname:
            resid_i = int(line_i[22:26])
            if resid_i in resids_done:
                continue
            residue_atoms = {}  # List of atom names & IDs for this residue
            for line_j in pdb_lines[i:]:
                # Make sure this is an atom line
                if not any([line_j.startswith(x) for x in ['ATOM', 'HETATM']]):
                    break
                # Make sure the following lines correspond to this resname and resid
                resid_j = int(line_j[22:26])
                if resid_j != resid_i or line_j[17:21].strip() != resname:
                    break
                # Read the atom data in for this residue
                atom_name = line_j[12:16].strip()
                atom_id = int(line_j[6:11])
                residue_atoms[atom_name] = atom_id
            # Add CONECT lines
            for bond in bond_list:
                conect_lines.append("CONECT{:>5d}{:>5d}\n".format(residue_atoms[bond[0]], residue_atoms[bond[1]]))
            resids_done.append(resid_i)

    # Write out the new PDB file, including CONECT lines
    with open(output, 'w') as f:
        for line in pdb_lines:
            if not line.startswith('END'):
                f.write(line)
            else:
                for line_c in conect_lines:
                    f.write(line_c)
                f.write(line)

    return None


def create_ligand_xml(prmtop, prepi, resname='LIG', output='lig.xml'):
    """
    Takes two AMBER parameter files (.prmtop and .prepi) for a small molecule and uses them to create an XML file
    which can be used to load the force field parameters for the ligand into OpenMM
    This function could do with some tidying at some point...

    Parameters
    ----------
    prmtop : str
        Name of the .prmtop file
    prepi : str
        Name of the .prepi file
    resname : str
        Residue name of the small molecule
    output : str
        Name of the output XML file
    """
    prmtop = parmed.load_file(prmtop)
    openmm_params = parmed.openmm.OpenMMParameterSet.from_structure(prmtop)
    tmp_xml = os.path.splitext(output)[0] + '-tmp.xml'
    openmm_params.write(tmp_xml)

    # Need to add some more changes here though, as the XML is incomplete
    atom_data, bond_list = read_prepi(prepi)

    # Read the temporary XML data back in
    with open(tmp_xml, 'r') as f:
        tmp_xml_lines = f.readlines()

    with open(output, 'w') as f:
        # First few lines get written out automatically
        for line in tmp_xml_lines[:5]:
            f.write(line)

        # First, we worry about the <AtomTypes> section
        f.write('  <AtomTypes>\n')
        for line in tmp_xml_lines:
            # Loop over the lines for each atom class
            if '<Type ' in line:
                # Read in the data for this XML line
                type_data = {}
                for x in line.split():
                    if '=' in x:
                        key = x.split('=')[0]
                        item = x.split('=')[1].strip('/>').strip('"')
                        type_data[key] = item

                # For each atom with this type, we write out a new line - can probably avoid doing this...
                for atom in atom_data:
                    if atom[1] != type_data['class']:
                        continue
                    new_line = '    <Type name="{}-{}" class="{}" element="{}" mass="{}"/>\n'.format(resname, atom[0],
                                                                                                     type_data['class'],
                                                                                                     type_data['element'],
                                                                                                     type_data['mass'])
                    f.write(new_line)
            elif '</AtomTypes>' in line:
                f.write('  </AtomTypes>\n')
                break

        # Now need to generate the actual residue template
        f.write(' <Residues>\n')
        f.write('  <Residue name="{}">\n'.format(resname))
        # First, write the atoms
        for atom in atom_data:
            f.write('   <Atom name="{0}" type="{1}-{0}" charge="{2}"/>\n'.format(atom[0], resname, atom[2]))
        # Then the bonds
        for bond in bond_list:
            f.write('   <Bond atomName1="{}" atomName2="{}"/>\n'.format(bond[0], bond[1]))
        f.write('  </Residue>\n')
        f.write(' </Residues>\n')

        # Now we can write out the rest, from the <HarmonicBondForce> section onwards
        for i, line_i in enumerate(tmp_xml_lines):
            if '<HarmonicBondForce>' in line_i:
                for line_j in tmp_xml_lines[i:]:
                    # Some lines need the typeX swapped for classX
                    f.write(line_j.replace('type', 'class'))
                break

    # Remove temporary file
    os.remove(tmp_xml)

    return None


def random_rotation_matrix():
    """
    Generate a random axis and angle for rotation of the water coordinates (using the
    method used for this in the ProtoMS source code (www.protoms.org), and then return
    a rotation matrix produced from these

    Returns
    -------
    rot_matrix : numpy.ndarray
        Rotation matrix generated
    """
    # First generate a random axis about which the rotation will occur
    rand1 = rand2 = 2.0

    while (rand1**2 + rand2**2) >= 1.0:
        rand1 = np.random.rand()
        rand2 = np.random.rand()
    rand_h = 2 * np.sqrt(1.0 - (rand1**2 + rand2**2))
    axis = np.array([rand1 * rand_h, rand2 * rand_h, 1 - 2*(rand1**2 + rand2**2)])
    axis /= np.linalg.norm(axis)

    # Get a random angle
    theta = np.pi * (2*np.random.rand() - 1.0)

    # Simplify products & generate matrix
    x, y, z = axis[0], axis[1], axis[2]
    x2, y2, z2 = axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]
    xy, xz, yz = axis[0]*axis[1], axis[0]*axis[2], axis[1]*axis[2]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[cos_theta + x2*(1-cos_theta),   xy*(1-cos_theta) - z*sin_theta, xz*(1-cos_theta) + y*sin_theta],
                           [xy*(1-cos_theta) + z*sin_theta, cos_theta + y2*(1-cos_theta),   yz*(1-cos_theta) - x*sin_theta],
                           [xz*(1-cos_theta) - y*sin_theta, yz*(1-cos_theta) + x*sin_theta, cos_theta + z2*(1-cos_theta)  ]])

    return rot_matrix


def shift_ghost_waters(ghost_file, topology=None, trajectory=None, t=None, output=None):
    """
    Translate all ghost waters in a trajectory out of the simulation box, to make
    visualisation clearer

    Parameters
    ----------
    ghost_file : str
        Name of the file containing the ghost water residue IDs at each frame
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Read in residue IDs for the ghost waters in each frame
    ghost_resids = read_ghosts_from_file(ghost_file)

    # Read in trajectory data
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Identify which atoms need to be moved out of sight
    ghost_atom_ids = []
    for frame in range(len(ghost_resids)):
        atom_ids = []
        for i, residue in enumerate(t.topology.residues):
            if i in ghost_resids[frame]:
                for atom in residue.atoms:
                    atom_ids.append(atom.index)
        ghost_atom_ids.append(atom_ids)

    # Shift coordinates of ghost atoms by several unit cells and write out trajectory
    for frame, atom_ids in enumerate(ghost_atom_ids):
        for index in atom_ids:
            t.xyz[frame, index, :] += 5 * t.unitcell_lengths[frame, :]

    # Either return the trajectory or save to file
    if output is None:
        return t
    else:
        t.save(output)
        return None


def wrap_waters(topology=None, trajectory=None, t=None, output=None):
    """
    Wrap water molecules if the coordinates haven't been wrapped by the DCDReporter

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    n_frames, n_atoms, n_dims = t.xyz.shape

    # Fix all frames
    for f in range(n_frames):
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if residue.name not in ['WAT', 'HOH']:
                continue

            # Find the maximum and minimum distances between this residue and the reference atom
            for atom in residue.atoms:
                if 'O' in atom.name:
                    pos = t.xyz[f, atom.index, :]

            # Calculate the correction vector based on the separation
            box = t.unitcell_lengths[f, :]

            new_pos = deepcopy(pos)
            for i in range(3):
                while new_pos[i] >= box[i]:
                    new_pos[i] -= box[i]
                while new_pos[i] <= 0:
                    new_pos[i] += box[i]

            correction = new_pos - pos

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def align_traj(topology=None, trajectory=None, t=None, reference=None, output=None):
    """
    Align a trajectory to the protein

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    reference : str
        Name of a PDB file to align the protein to. May be better to visualise
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory data, if not already
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)

    # Align trajectory based on protein C-alpha atoms
    protein_ids = [atom.index for atom in t.topology.atoms if atom.residue.is_protein and atom.name == 'CA']
    if reference is None:
        # If there is no reference then align to the first frame in the trajectory
        t.superpose(t, atom_indices=protein_ids)
    else:
        # Load a reference PDB to align the structure to
        t_ref = mdtraj.load(reference)
        t.superpose(t_ref, atom_indices=protein_ids)

    # Return or save trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def recentre_traj(topology=None, trajectory=None, t=None, name='CA', resname='ALA', resid=1, output=None):
    """
    Recentre a trajectory based on a specific protein residue. Assumes that the
    protein has not been broken by periodic boundaries.
    Would be best to do this step before aligning a trajectory

    Parameters
    ----------
    topology : str
        Name of the topology/connectivity file (e.g. PDB, GRO, etc.)
    trajectory : str
        Name of the trajectory file (e.g. DCD, XTC, etc.)
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    resname : str
        Name of the atom to centre the trajectorname.lower(
    resname : str
        Name of the protein residue to centre the trajectory on. Should be a
        binding site residue
    resid : int
        ID of the protein residue to centre the trajectory. Should be a binding
        site residue
    output : str
        Name of the file to which the new trajectory is written. If None, then a
        Trajectory will be returned

    Returns
    -------
    t : mdtraj.Trajectory
        Will return a trajectory object, if no output file name is given
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get IDs of protein atoms
    protein_ids = [atom.index for atom in t.topology.atoms if atom.residue.is_protein]

    # Find the index of the C-alpha atom of this residue
    ref_idx = None
    for residue in t.topology.residues:
        if residue.name.lower() == resname.lower() and residue.resSeq == resid:
            for atom in residue.atoms:
                if atom.name.lower() == name.lower():
                    ref_idx = atom.index
    if ref_idx is None:
        raise Exception("Could not find atom {} of residue {}{}!".format(name, resname.capitalize(), resid))

    # Fix all frames
    for f in range(n_frames):
        # Box dimensions for this frame
        box = t.unitcell_lengths[f, :]

        # Recentre all protein chains
        for chain in t.topology.chains:
            # Skip if this is a non-protein chain
            if not all([atom.index in protein_ids for atom in chain.atoms]):
                continue

            # Find the closest distance between this chain and the reference
            min_dists = 1e8 * np.ones(3)
            for atom in chain.atoms:
                # Distance between this atom and reference
                v = t.xyz[f, atom.index, :] - t.xyz[f, ref_idx, :]
                for i in range(3):
                    if abs(v[i]) < min_dists[i]:
                        min_dists[i] = v[i]

            # Calculate the correction vector based on the separation
            correction = np.zeros(3)

            for i in range(3):
                if -2 * box[i] < min_dists[i] < -0.5 * box[i]:
                    correction[i] += box[i]
                elif 0.5 * box[i] < min_dists[i] < 2 * box[i]:
                    correction[i] -= box[i]

            # Apply the correction vector to each atom in the residue
            for atom in chain.atoms:
                t.xyz[f, atom.index, :] += correction

        # Recentre all non-protein residues
        for residue in t.topology.residues:
            # Skip if this is a protein residue
            if any([atom.index in protein_ids for atom in residue.atoms]):
                continue

            # Find the closest distance between this residue and the reference
            min_dists = 1e8 * np.ones(3)
            for atom in residue.atoms:
                # Distance between this atom and reference
                v = t.xyz[f, atom.index, :] - t.xyz[f, ref_idx, :]
                for i in range(3):
                    if abs(v[i]) < min_dists[i]:
                        min_dists[i] = v[i]

            # Calculate the correction vector based on the separation
            correction = np.zeros(3)

            for i in range(3):
                if -2 * box[i] < min_dists[i] < -0.5 * box[i]:
                    correction[i] += box[i]
                elif 0.5 * box[i] < min_dists[i] < 2 * box[i]:
                    correction[i] -= box[i]

            # Apply the correction vector to each atom in the residue
            for atom in residue.atoms:
                t.xyz[f, atom.index, :] += correction

    # Either return or save the trajectory
    if output is None:
        return t
    else:
        t.save(output)
        return None


def write_sphere_traj(radius, ref_atoms=None, topology=None, trajectory=None, t=None, sphere_centre=None,
                      output='gcmc_sphere.pdb', initial_frame=False):
    """
    Write out a multi-frame PDB file containing the centre of the GCMC sphere

    Parameters
    ----------
    radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere (list of dictionaries)
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    t : mdtraj.Trajectory
        Trajectory object, if already loaded
    sphere_centre : simtk.unit.Quantity
        Coordinates around which the GCMC sphere is based
    output : str
        Name of the output PDB file
    initial_frame : bool
        Write an extra frame for the topology at the beginning of the trajectory.
        Sometimes necessary when visualising a trajectory loaded onto a PDB
    """
    # Load trajectory
    if t is None:
        t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            if 'chain' not in ref_atom.keys():
                warnings.warn("Chains are not specified for at least one reference atom, will select the first instance"
                              " of {}{} found".format(ref_atom['resname'].capitalize(), ref_atom['resid']))
                for residue in t.topology.residues:
                    if residue.name == ref_atom['resname'] and str(residue.resSeq) == ref_atom['resid']:
                        for atom in residue.atoms:
                            if atom.name == ref_atom['name']:
                                ref_indices.append(atom.index)
                                found = True
            else:
                if type(ref_atom['chain']) == str:
                    ref_atom['chain'] = ord(ref_atom['chain'])-65
                for residue in t.topology.residues:
                    if residue.name == ref_atom['resname'] and \
                            str(residue.resSeq) == ref_atom['resid'] and \
                            residue.chain.index == ref_atom['chain']:
                        for atom in residue.atoms:
                            if atom.name == ref_atom['name']:
                                ref_indices.append(atom.index)
                                found = True
            if not found:
                raise Exception("Atom {} of residue {}{} and chain {} not found!".format(ref_atom['name'],
                                                                                         ref_atom['resname'].capitalize(),
                                                                                         ref_atom['resid'],
                                                                                         chr(ref_atom['chain']+65)))


    # Loop over all frames and write to PDB file
    with open(output, 'w') as f:
        f.write("HEADER GCMC SPHERE\n")
        f.write("REMARK RADIUS = {} ANGSTROMS\n".format(radius))

        # Figure out the initial coordinates if requested
        if initial_frame:
            t_i = mdtraj.load(topology, discard_overlapping_frames=False)
            # Calculate centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t_i.xyz[0, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL\n")
            f.write("HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(1, 'CTR', 'SPH', 1,
                                                                                             centre[0], centre[1],
                                                                                             centre[2]))
            f.write("ENDMDL\n")

        # Loop over all frames
        for frame in range(n_frames):
            # Calculate sphere centre
            if sphere_centre is None:
                centre = np.zeros(3)
                for idx in ref_indices:
                    centre += t.xyz[frame, idx, :]
                centre *= 10 / len(ref_indices)  # Also convert from nm to A
            else:
                centre = sphere_centre.in_units_of(unit.angstroms)._value
            # Write to PDB file
            f.write("MODEL {}\n".format(frame+1))
            f.write("HETATM{:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}\n".format(1, 'CTR', 'SPH', 1,
                                                                                             centre[0], centre[1],
                                                                                             centre[2]))
            f.write("ENDMDL\n")

    return None

def cluster_waters(topology, trajectory, sphere_radius, ref_atoms=None, sphere_centre=None, cutoff=2.4,
                   output='gcmc_clusts.pdb'):
    """
    Carry out a clustering analysis on GCMC water molecules with the sphere. Based on the clustering
    code in the ProtoMS software package.

    This function currently assumes that the system has been aligned and centred on the GCMC sphere (approximately).

    Parameters
    ----------
    topology : str
        Topology of the system, such as a PDB file
    trajectory : str
        Trajectory file, such as DCD
    sphere_radius : float
        Radius of the GCMC sphere in Angstroms
    ref_atoms : list
        List of reference atoms for the GCMC sphere (list of dictionaries)
    sphere_centre : simtk.unit.Quantity
        Coordinates around which the GCMC sphere is based
    cutoff : float
        Distance cutoff used in the clustering
    output : str
        Name of the output PDB file containing the clusters
    """
    # Load trajectory
    t = mdtraj.load(trajectory, top=topology, discard_overlapping_frames=False)
    n_frames, n_atoms, n_dims = t.xyz.shape
    pdb = app.pdbfile.PDBFile(topology)

    # Get reference atom IDs
    if ref_atoms is not None:
        ref_indices = []
        for ref_atom in ref_atoms:
            found = False
            if 'chain' not in ref_atom.keys():
                warnings.warn("Chains are not specified for at least one reference atom, will select the first instance"
                              " of {}{} found".format(ref_atom['resname'].capitalize(), ref_atom['resid']))
                for residue in t.topology.residues:
                    if residue.name == ref_atom['resname'] and str(residue.resSeq) == ref_atom['resid']:
                        for atom in residue.atoms:
                            if atom.name == ref_atom['name']:
                                ref_indices.append(atom.index)
                                found = True
            else:
                if type(ref_atom['chain']) == str:
                    ref_atom['chain'] = ord(ref_atom['chain'])-65
                for residue in t.topology.residues:
                    if residue.name == ref_atom['resname'] and \
                            str(residue.resSeq) == ref_atom['resid'] and \
                            residue.chain.index == ref_atom['chain']:
                        for atom in residue.atoms:
                            if atom.name == ref_atom['name']:
                                ref_indices.append(atom.index)
                                found = True
            if not found:
                raise Exception("Atom {} of residue {}{} and chain {} not found!".format(ref_atom['name'],
                                                                                         ref_atom['resname'].capitalize(),
                                                                                         ref_atom['resid'],
                                                                                         chr(ref_atom['chain']+65)))

    wat_coords = []  # Store a list of water coordinates
    wat_frames = []  # Store a list of the frame that each water is in

    # Get list of water oxygen atom IDs
    wat_ox_ids = []
    for residue in t.topology.residues:
        if residue.name.lower() in ['wat', 'hoh']:
            for atom in residue.atoms:
                if atom.name.lower() == 'o':
                    wat_ox_ids.append(atom.index)

    # Get the coordinates of all GCMC water oxygen atoms
    for f in range(n_frames):

        # Calculate sphere centre for this frame
        if ref_atoms is not None:
            centre = np.zeros(3)
            for idx in ref_indices:
                centre += t.xyz[f, idx, :]
            centre /= len(ref_indices)
        else:
            centre = sphere_centre.in_units_of(unit.nanometer)._value

        # For all waters, check the distance to the sphere centre
        for o in wat_ox_ids:
            # Calculate PBC-corrected vector
            vector = t.xyz[f, o, :] - centre

            # Check length and add to list if within sphere
            if 10*np.linalg.norm(vector) <= sphere_radius:  # *10 to give Angstroms
                wat_coords.append(10 * t.xyz[f, o, :])  # Convert to Angstroms
                wat_frames.append(f)

    # Calculate water-water distances - if the waters are in the same frame are assigned a very large distance
    dist_list = []
    for i in range(len(wat_coords)):
        for j in range(i+1, len(wat_coords)):
            if wat_frames[i] == wat_frames[j]:
                dist = 1e8
            else:
                dist = np.linalg.norm(wat_coords[i] - wat_coords[j])
            dist_list.append(dist)

    # Cluster the waters hierarchically
    tree = hierarchy.linkage(dist_list, method='average')
    wat_clust_ids = hierarchy.fcluster(tree, t=cutoff, criterion='distance')
    n_clusts = max(wat_clust_ids)

    # Sort the clusters by occupancy
    clusts = []
    for i in range(1, n_clusts+1):
        occ = len([wat for wat in wat_clust_ids if wat == i])
        clusts.append([i, occ])
    clusts = sorted(clusts, key=lambda x: -x[1])
    clust_ids_sorted = [x[0] for x in clusts]
    clust_occs_sorted = [x[1] for x in clusts]

    # Calculate the cluster centre and representative position for each cluster
    rep_coords = []
    for i in range(n_clusts):
        clust_id = clust_ids_sorted[i]
        # Calculate the mean position of the cluster
        clust_centre = np.zeros(3)
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                clust_centre += wat_coords[j]
        clust_centre /= clust_occs_sorted[i]

        # Find the water observation which is closest to the mean position
        min_dist = 1e8
        rep_wat = None
        for j, wat in enumerate(wat_clust_ids):
            if wat == clust_id:
                dist = np.linalg.norm(wat_coords[j] - clust_centre)
                if dist < min_dist:
                    min_dist = dist
                    rep_wat = j
        rep_coords.append(wat_coords[rep_wat])

    # Write the cluster coordinates to a PDB file
    with open(output, 'w') as f:
        f.write("REMARK Clustered GCMC water positions written by grand\n")
        for i in range(n_clusts):
            coords = rep_coords[i]
            occ1 = clust_occs_sorted[i]
            occ2 = occ1 / float(n_frames)
            f.write("ATOM  {:>5d} {:<4s} {:<4s} {:>4d}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}\n".format(1, 'O',
                                                                                                             'WAT', i+1,
                                                                                                             coords[0],
                                                                                                             coords[1],
                                                                                                             coords[2],
                                                                                                             occ2, occ2))
            f.write("TER\n")
        f.write("END")

    return None

