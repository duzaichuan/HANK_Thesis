# Replication files for 2020 HANK model Thesis

Code that solves and analyzes the HANK model developed for my 2020 thesis in economics "Heterogeneous Agents and Unemployment in a New Keynesian General Equilibrium Model". 

The files CH5_-CH7 conducts the analyzes in the associated chapter of the thesis, drawing on the steady state and model blocks in HANKSAM.py. 
The files FigUtils.py and Utils2.py contains various utility functions that the main files use. 

Many of the files (asymptotic.py, determinacy.py, het_block.py, jacobian.py, nonlinear.py, simple_block.py, solved_block-py, utils.py) comes from the SHADE-files (github: https://github.com/shade-econ/sequence-jacobian/#sequence-space-jacobian) required to compute impulse responses using the sequence-space jacobian method. 
