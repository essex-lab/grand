[![Documentation Status](https://readthedocs.org/projects/grand/badge/?version=latest)](https://grand.readthedocs.io/en/latest/?badge=latest)

# _grand_ : Grand Canonical Water Sampling in OpenMM

Version : 1.0.0

### Background

This Python module is designed to be run with OpenMM in order to simulate grand
canonical Monte Carlo (GCMC) insertion and deletion moves of water molecules.
This allows the particle number to vary according to a fixed chemical
potential, and offers enhanced sampling of water molecules in occluded
binding sites.
The theory behind our work on GCMC sampling can be found in the References
section below.

### Installation & Usage

This module can be installed from this directory by running the following
command:

```commandline
python setup.py install
```

The dependencies of this module can be installed as:

```commandline
conda install -c omnia openmm mdtraj parmed openmmtools pymbar
pip install lxml
```

Several (very short) examples of how this module is ran alongside OpenMM can be found in
the `examples/` directory.
Additional [examples](https://github.com/essex-lab/grand-paper) and 
[documentation](https://grand.readthedocs.io/en/latest/) are also available.

The _grand_ module is released under the MIT licence. If results from this
module contribute to a publication, we only ask that you cite the
publications below.

### Contributors

- Marley Samways `<mls2g13@soton.ac.uk>`
- Hannah Bruce Macdonald `<hannah.brucemacdonald@choderalab.org>`

### Contact

If you have any problems or questions regarding this module, please contact
one of the contributors, or send an email to `<jessexgroup@gmail.com>`.

### References

1. G. A. Ross, M. S. Bodnarchuk, J. W. Essex, _J. Am. Chem. Soc._ 2015, 
137, 47, 14930-14943, DOI: https://doi.org/10.1021/jacs.5b07940
2. G. A. Ross, H. E. Bruce Macdonald, C. Cave-Ayland, A. I. Cabedo
Martinez, J. W. Essex, _J. Chem. Theory Comput._ 2017, 13, 12, 6373-6381, DOI:
https://doi.org/10.1021/acs.jctc.7b00738
