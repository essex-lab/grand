"""
Setup script to facilitate installation of the GCMC OpenMM scripts

Marley L. Samways
"""

import os
from setuptools import setup


# Read version number from the __init__.py file
with open(os.path.join('grand', '__init__.py'), 'r') as f:
    for line in f.readlines():
        if '__version__' in line:
            version = line.split()[-1].strip('"')

setup(name="grand",
      version=version,
      description="OpenMM-based implementation of grand canonical Monte Carlo (GCMC)",
      author="Marley L. Samways",
      author_email="mls2g13@soton.ac.uk",
      packages=["grand", "grand.tests"],
      #install_requires=["numpy", "mdtraj"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      test_suite="grand.tests",
      package_data={"grand": ["data/*", "data/tests/*"]}
      )

