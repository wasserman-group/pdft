PDFT 
====

Partition Density Functional Theory: A density functional toolbox oriented to fragment calculations.


[//]: # (Badges)
[![Build Status](https://github.com/wasserman-group/pdft/workflows/CI/badge.svg)](https://github.com/wasserman-group/pdft/actions)

[![Build Status](https://travis-ci.com/vhchavez/pdft.svg?token=qfJoUsJ2RErCUYXqxfAQ&branch=master)](https://travis-ci.com/vhchavez/pdft)
[![codecov](https://codecov.io/gh/vhchavez/pdft/branch/master/graph/badge.svg?token=83bevc0aMc)](https://codecov.io/gh/vhchavez/pdft)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/VHchavez/pdft.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/VHchavez/pdft/context:python)

PDFT is designed for calculating the ground-state properties of a full molecular system via self-consistent calculations on isolated fragments.

### Overview:
PDFT can perform ground-state density functional theory calculations.  
Contains density-to-potential inversion procedures.  
Allows for the extraction of density functional approximations' ingredients such as density, grad(density) and laplacian(density), and kinetic energy density on the grid.  
Written in Python using PSIAPI which allows for rapid prototyping and testing of embedding methods and density functional approximations.  

### Installation:
The use of conda is highly recommended. 
1) Install Psi4:
```
conda install -c psi4 psi4
```
To create conda environment with psi4:
```
conda create -n pdft -c psi4 psi4 python=3.8
```

2) Install PDFT:
```
git clone https://github.com/VHchavez/pdft.git
cd pdft
pip install .
```

Support for pip install and conda install coming soon!

### Tutorials:
Binder examples coming soon!

<br>
<br>

#### Copyright

Copyright (c) 2020, The Wasserman Group

#### Acknowledgements
Project based on the [MolSSi Cookiecutter](https://github.com/molssi/cookiecutter-cms).  
Victor H. Chavez was supported by a fellowship from The Molecular Sciences Software Institute under NSF grant OAC-1547580.  

