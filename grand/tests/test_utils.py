"""
Description
-----------
This file contains functions written to test the functions in the grand.utils sub-module

Marley Samways
"""

import os
import unittest
import numpy as np
import mdtraj
from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *
from grand import utils


outdir = os.path.join(os.path.dirname(__file__), 'output', 'utils')


class TestUtils(unittest.TestCase):
    """
    Class to store the tests for grand.utils
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

    def test_PDBRestartReporter(self):
        """
        Check that the PDBRestart Reporter works okay
        """
        # First need to create a Simulation
        # Load a pre-equilibrated water box
        pdb = PDBFile(utils.get_data_file(os.path.join('tests', 'water_box-eq.pdb')))

        # Set up system
        ff = ForceField("tip3p.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=12.0 * angstroms,
                                 constraints=HBonds, switchDistance=10 * angstroms)

        integrator = LangevinIntegrator(300 * kelvin, 1.0 / picosecond, 0.002 * picoseconds)

        try:
            platform = Platform.getPlatformByName('CUDA')
        except:
            try:
                platform = Platform.getPlatformByName('OpenCL')
            except:
                platform = Platform.getPlatformByName('CPU')

        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(300 * kelvin)
        simulation.context.setPeriodicBoxVectors(*pdb.topology.getPeriodicBoxVectors())

        # Initialise the reporter
        restart_file = os.path.join(outdir, 'restart.pdb')
        pdb_reporter = utils.PDBRestartReporter(filename=restart_file,
                                            topology=pdb.topology)
        # Write a restart PDB out
        pdb_reporter.report(simulation, simulation.context.getState(getPositions=True))

        # Try to load the PDB to check that it exists and is valid
        pdb_out = PDBFile(restart_file)

        return None

    def test_get_data_file(self):
        """
        Test the get_data_file function, designed to retrieve certain package data files
        """
        # Check that a known file is returned
        assert os.path.isfile(utils.get_data_file('tip3p.pdb'))
        # Check that a made up file raises an exception
        self.assertRaises(Exception, lambda: utils.get_data_file('imaginary.file'))

        return None

    def test_add_remove_ghosts(self):
        """
        Test that the add_ghosts() and remove_ghosts() functions work fine
        """
        # Need to load some test data
        bpti_orig = PDBFile(utils.get_data_file(os.path.join('tests', 'bpti.pdb')))
        # Count the number of water molecules in this file
        n_wats_orig = 0
        for residue in bpti_orig.topology.residues():
            if residue.name in ["HOH", "WAT"]:
                n_wats_orig += 1
        # Run add_ghosts_function
        n_add = 5  # Add this number of waters
        new_file = os.path.join(outdir, 'bpti-ghosts-added.pdb')
        # Make sure this file doesn't already exist
        if os.path.isfile(new_file):
            os.remove(new_file)

        # Add ghosts to the topology
        topology, positions, ghosts = utils.add_ghosts(topology=bpti_orig.topology, positions=bpti_orig.positions,
                                                       n=n_add, pdb=new_file)
        # Make sure that the number of ghost molecule IDs is correct
        assert len(ghosts) == n_add

        # Count the number of water molecules in the topology and check it's right
        n_wats_add = 0
        for residue in topology.residues():
            if residue.name in ["HOH", "WAT"]:
                n_wats_add += 1
        assert n_wats_add == n_wats_orig + n_add

        # Make sure that the new file was created
        assert os.path.isfile(new_file)

        # Now remove the ghosts
        new_file2 = os.path.join(outdir, 'bpti-ghosts-removed.pdb')
        # Make sure this file doesn't already exist
        if os.path.isfile(new_file2):
            os.remove(new_file2)

        # Run the remove_ghosts function
        topology, positions = utils.remove_ghosts(topology, positions, ghosts=ghosts, pdb=new_file2)

        # Count the number of water molecules in the topology and check it's right
        n_wats_remove = 0
        for residue in topology.residues():
            if residue.name in ["HOH", "WAT"]:
                n_wats_remove += 1
        assert n_wats_remove == n_wats_orig

        # Make sure that the new file was created
        assert os.path.isfile(new_file2)

        return None

    def test_read_prepi(selfs):
        """
        Test the read_prepi() function
        """
        # Using benzene as an example
        benzene_atoms = ["C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4", "H5", "H6"]
        benzene_bonds = [["C1", "C2"], ["C2", "C5"], ["C5", "C6"], ["C6", "C4"], ["C4", "C3"], ["C3", "C1"],
                         ["C1", "H1"], ["C2", "H2"], ["C3", "H3"], ["C4", "H4"], ["C5", "H5"], ["C6", "H6"]]

        # Read in a benzene prepi
        atom_data, prepi_bonds = utils.read_prepi(utils.get_data_file(os.path.join('tests', 'benzene.prepi')))

        # Make sure the right number of atoms and bonds have been read in
        assert len(atom_data) == len(benzene_atoms)
        assert len(prepi_bonds) == len(benzene_bonds)

        # Check all atoms are accounted for
        assert all([any([atom[0] == name for atom in atom_data]) for name in benzene_atoms])

        # Make sure the atoms have the correct types
        assert all([atom[1] == 'ca' for atom in atom_data if atom[0].startswith('C')])
        assert all([atom[1] == 'ha' for atom in atom_data if atom[0].startswith('H')])

        # Make sure charges have been read in for all atoms (check that they can be floats)
        charges = [float(atom[2]) for atom in atom_data]

        # Make sure all bonds have been accounted for
        assert all([any([bond1[0] in bond2 and bond1[1] in bond2 for bond2 in prepi_bonds]) for bond1 in benzene_bonds])

        return None

    def test_write_conect(self):
        """
        Test the write_conect() function
        """
        # Using benzene as an example
        benzene_bonds = [["C1", "C2"], ["C2", "C5"], ["C5", "C6"], ["C6", "C4"], ["C4", "C3"], ["C3", "C1"],
                         ["C1", "H1"], ["C2", "H2"], ["C3", "H3"], ["C4", "H4"], ["C5", "H5"], ["C6", "H6"]]

        # Write a PDB file with CONECT lines
        utils.write_conect(pdb=utils.get_data_file(os.path.join('tests', 'benzene.pdb')), resname="BEN",
                           prepi=utils.get_data_file(os.path.join('tests', 'benzene.prepi')),
                           output=os.path.join(outdir, 'benzene-conect.pdb'))

        # Read in atom IDs
        atom_dict = {}
        conect_bonds = []
        with open(os.path.join(outdir, 'benzene-conect.pdb'), 'r') as f:
            for line in f.readlines():
                # Read in atom info
                if line.startswith('HETATM'):
                    index = int(line[6:11])
                    name = line[12:16].strip()
                    atom_dict[index] = name
                # Read CONECT lines
                if line.startswith('CONECT'):
                    index1 = int(line.split()[1])
                    index2 = int(line.split()[2])
                    conect_bonds.append([atom_dict[index1], atom_dict[index2]])

        # Check that the bonds are the same
        assert len(conect_bonds) == len(benzene_bonds)
        assert all([any([bond1[0] in bond2 and bond1[1] in bond2 for bond2 in conect_bonds]) for bond1 in benzene_bonds])

        return None

    def test_create_ligand_xml(self):
        """
        Test the create_ligand_xml() function
        """
        # Using benzene as an example
        benzene_atoms = ["C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4", "H5", "H6"]
        benzene_bonds = [["C1", "C2"], ["C2", "C5"], ["C5", "C6"], ["C6", "C4"], ["C4", "C3"], ["C3", "C1"],
                         ["C1", "H1"], ["C2", "H2"], ["C3", "H3"], ["C4", "H4"], ["C5", "H5"], ["C6", "H6"]]

        # Create the XML file
        xml_file = os.path.join(outdir, 'benzene.xml')
        utils.create_ligand_xml(prmtop=utils.get_data_file(os.path.join('tests', 'benzene.prmtop')),
                                prepi=utils.get_data_file(os.path.join('tests', 'benzene.prepi')),
                                resname='BEN', output=xml_file)

        # Check that the file exists
        assert os.path.isfile(xml_file)

        # Briefly check the contents of the file...
        with open(xml_file, 'r') as f:
            all_lines = f.readlines()
        atom_type_lines = [line for line in all_lines if "<Type " in line]
        atom_name_lines = [line for line in all_lines if "<Atom name=" in line]
        bond_lines = [line for line in all_lines if "<Bond atomName1=" in line]

        # Check atom types are correct
        assert all(['ca' in line for line in atom_type_lines if 'BEN-C' in line])
        assert all(['ha' in line for line in atom_type_lines if 'BEN-H' in line])

        # Check all atoms are accounted for
        assert len(atom_name_lines) == 12
        assert all([any([name in line for line in atom_name_lines]) for name in benzene_atoms])

        # Check that the bonds are correct
        assert len(bond_lines) == len(benzene_bonds)
        assert all([any([bond[0] in line and bond[1] in line for line in bond_lines]) for bond in benzene_bonds])

        return None

    def test_rotation_matrix(self):
        """
        Test that the random_rotation_matrix() function works as expected
        """
        R = utils.random_rotation_matrix()
        # Matrix must be 3x3
        assert R.shape == (3, 3)
        # Make sure that det(R) is 1
        assert np.isclose(np.linalg.det(R), 1.0)
        # Check that the inverse is equal to the transpose of R
        assert np.all(np.isclose(np.linalg.inv(R), R.T))
        # Make sure that a different matrix is returned each time
        assert not np.all(np.isclose(R, utils.random_rotation_matrix()))

        return None

    def test_shift_ghost_waters(self):
        """
        Test that the shift_ghost_waters() function works
        """
        # First check that the function can return Trajectory if required
        t = utils.shift_ghost_waters(ghost_file=utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')),
                                     topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                     trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t, mdtraj.Trajectory)
        # Then make sure that the code can also write a DCD file
        utils.shift_ghost_waters(ghost_file=utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')),
                                 topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                 trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                                 output=os.path.join(outdir, 'bpti-shifted.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-shifted.dcd'))

        # Need to check that the ghost residues are out of the simulation box
        ghost_waters = []
        with open(utils.get_data_file(os.path.join('tests', 'bpti-ghost-wats.txt')), 'r') as f:
            for line in f.readlines():
                ghost_waters.append([int(resid) for resid in line.split(',')])

        # Now want to check that there are no ghost water molecules inside the box
        n_frames, _, _ = t.xyz.shape
        failed_checks = []  # List of all checks done - True if the water is too close to the box
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for resid, residue in enumerate(t.topology.residues):
                # Only consider ghost waters
                if resid not in ghost_waters[f]:
                    continue

                for atom in residue.atoms:
                    if 'O' in atom.name:
                        pos = t.xyz[f, atom.index, :]

                # Check if this water is too close to the box (within 2 box lengths) in any dimension
                for i in range(3):
                    failed_checks.append(pos[i] < 2 * box[i])
        # Make sure that none of the waters are ever too close
        assert not any(failed_checks)

        return None

    def test_wrap_waters(self):
        """
        Test that the wrap_waters() function works as desired
        """
        # First check that the function can return Trajectory if required
        t = utils.wrap_waters(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t, mdtraj.Trajectory)
        # Then make sure that the code can also write a DCD file
        utils.wrap_waters(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                          trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                          output=os.path.join(outdir, 'bpti-wrapped.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-wrapped.dcd'))

        # Now want to check that there are no water molecules outside the box
        n_frames, _, _ = t.xyz.shape
        failed_checks = []  # List of all checks done - True if the water is outside the box
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for residue in t.topology.residues:
                if residue.name not in ['WAT', 'HOH']:
                    continue

                for atom in residue.atoms:
                    if 'O' in atom.name:
                        pos = t.xyz[f, atom.index, :]

                # Check if this water is outside the box in any dimension
                for i in range(3):
                    failed_checks.append(pos[i] < 0 or pos[i] > box[i])
        # Make sure that none of the waters are ever outside
        assert not any(failed_checks)

        return None

    def test_align_traj(self):
        """
        Test that the align_traj() function works
        Hard to test this properly, so will just make sure that the function runs without errors
        """
        # First check that the function can return Trajectory if required (with a reference)
        t1 = utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                              reference=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')))
        assert isinstance(t1, mdtraj.Trajectory)

        # Now check without a reference
        t2 = utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                              trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')))
        assert isinstance(t2, mdtraj.Trajectory)

        # Then make sure that the code can also write a DCD file
        utils.align_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                         trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                         output=os.path.join(outdir, 'bpti-aligned.dcd'))
        assert os.path.isfile(os.path.join(outdir, 'bpti-aligned.dcd'))

        return None

    def test_recentre_traj(self):
        """
        Test that the recentre_traj() function works
        """
        pdb = utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb'))
        dcd = utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd'))
        out = os.path.join(outdir, 'bpti-recentred.dcd')
        # Check that there is an Exception if the reference residue doesn't exist (His1 doesn't for BPTI)
        self.assertRaises(Exception, lambda: utils.recentre_traj(topology=pdb, trajectory=dcd, resname='HIS',
                                                                 resid=1))
        # Make sure the function isn't case-sensitive to the residue name
        utils.recentre_traj(topology=pdb, trajectory=dcd, resname='Arg', resid=1, output=out)
        os.remove(out)

        # Make sure that a trajectory object can be returned
        t = utils.recentre_traj(topology=pdb, trajectory=dcd, resname='TYR', resid=10)
        assert isinstance(t, mdtraj.Trajectory)
        # Make sure that a file can be written out
        utils.recentre_traj(topology=pdb, trajectory=dcd, resname='TYR', resid=10, output=out)
        assert os.path.isfile(out)

        # Make sure this residue is at the centre of the trajectory
        n_frames, _, _ = t.xyz.shape
        for residue in t.topology.residues:
            if residue.name == 'TYR' and residue.resSeq == 10:
                for atom in residue.atoms:
                    if atom.name.lower() == 'ca':
                        ref = atom.index

        # Check all residues and make sure that at least one atom is within 0.5L of the reference atom in each frame
        failed_checks = []  # List of failed checks - True is that the COG is too far from the centre
        for f in range(n_frames):
            box = t.unitcell_lengths[f, :]

            for residue in t.topology.residues:
                # Ignore protein
                if residue.is_protein:
                    continue

                # Calculate centre of geometry
                cog = np.zeros(3)
                for atom in residue.atoms:
                    cog += t.xyz[f, atom.index, :]
                cog /= residue.n_atoms

                # Check distance from the reference
                dists_x = [abs(t.xyz[f, atom.index, 0] - t.xyz[f, ref, 0]) for atom in residue.atoms]
                dists_y = [abs(t.xyz[f, atom.index, 1] - t.xyz[f, ref, 1]) for atom in residue.atoms]
                dists_z = [abs(t.xyz[f, atom.index, 2] - t.xyz[f, ref, 2]) for atom in residue.atoms]

                # Check the closest distance for this residue in each dimension
                failed_checks.append(min(dists_x) > 0.5 * box[0])
                failed_checks.append(min(dists_y) > 0.5 * box[1])
                failed_checks.append(min(dists_z) > 0.5 * box[2])

        assert not any(failed_checks)

        return None

    def test_write_sphere_traj(self):
        """
        Test that the write_sphere_traj() function works
        May need to make this function more thorough...
        """
        pdb = utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb'))
        dcd = utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd'))
        out = os.path.join(outdir, 'gcmc_sphere-bpti.pdb')
        radius = 4.0
        # Make sure that the function doesn't work if an incorrect reference is given (should be Tyr10 and Asn43)
        ref_atoms_wrong = [{'name': 'CA', 'resname': 'TYR', 'resid': '9'},
                           {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]
        self.assertRaises(Exception, lambda: utils.write_sphere_traj(ref_atoms=ref_atoms_wrong,
                                                                     radius=radius, topology=pdb, trajectory=dcd))
        # Write a sphere PDB
        ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                     {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]
        utils.write_sphere_traj(ref_atoms=ref_atoms, radius=radius, topology=pdb,
                                trajectory=dcd, output=out, initial_frame=True)
        assert os.path.isfile(out)

        # Make sure the sphere radius is
        with open(out, 'r') as f:
            for line in f.readlines():
                if line.startswith('REMARK RADIUS'):
                    r = float(line.split()[3])
                    break
        assert np.isclose(radius, r)

        return None

    def test_cluster_waters(self):
        """
        Test the cluster_waters() function
        """
        # Need a centred & aligned trajectory first
        dcd = os.path.join(outdir, 'bpti-centred-aligned.dcd')
        t = utils.recentre_traj(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                                trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                                resname='TYR', resid=10)
        utils.align_traj(t=t, output=dcd)

        # Run the clustering function
        clust_pdb = os.path.join(outdir, 'bpti-clusters.pdb')
        ref_atoms = [{'name': 'CA', 'resname': 'TYR', 'resid': '10'},
                     {'name': 'CA', 'resname': 'ASN', 'resid': '43'}]
        utils.cluster_waters(topology=utils.get_data_file(os.path.join('tests', 'bpti-ghosts.pdb')),
                             trajectory=utils.get_data_file(os.path.join('tests', 'bpti-raw.dcd')),
                             sphere_radius=4.0, ref_atoms=ref_atoms, output=clust_pdb)

        # Check that the file exists
        assert os.path.isfile(clust_pdb)

        # Test that the PDB can be opened - this checks that the formatting is correct
        pdb = PDBFile(clust_pdb)

        # Read in the occupancies of each cluster
        occupancies = []
        with open(clust_pdb, 'r') as f:
            for line in f.readlines():
                if line.startswith('ATOM'):
                    occupancies.append(float(line[54:60]))

        # Make sure all occupancies are between 0 and 1
        assert all([0.0 <= occ <= 1.0 for occ in occupancies])

        return None
