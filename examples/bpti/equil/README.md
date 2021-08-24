# Example equilibration and production for BPTI

The `equil/` directory contains the necessary scripts and co-ordinate files to run a typical GCMC/MD equilibration protocol on a protein system.

The `prod/` directory contains an example script to run a production simulation on the equilibrated BPTI structure. The script also includes functions to post-process the trajectory and perform a clustering analysis on the water locations. The `bpti-restart.py` script demonstrates how to continue a simulation using the `.rst7` file.

To run the equilibration and production simulations, the following commands are required:
```commandline
cd equil/
python equil-uvt1.py
python equil-npt.py
python equil-uvt2.py
cd ../prod/
python bpti.py
```
