# dolfin_navier_scipy

[![DOI](https://zenodo.org/badge/15728657.svg)](https://zenodo.org/badge/latestdoi/15728657)
[![PyPI version](https://badge.fury.io/py/dolfin-navier-scipy.png)](https://badge.fury.io/py/dolfin-navier-scipy)
[![Documentation Status](https://readthedocs.org/projects/dolfin-navier-scipy/badge/?version=latest)](https://dolfin-navier-scipy.readthedocs.io/en/latest/?badge=latest)

This python module `dns` provides an interface between the FEM toolbox [`FEniCS`](www.fenicsproject.org) and [`SciPy`](www.scipy.org) in view of simulation and control of incompressible flows. Basically, `FEniCS` is used to discretize the *incompressible Navier-Stokes equations* in space. Then `dns` makes the discretized operators available in `SciPy` for use in model reduction, simulation, or control and optimization. 

`dns` also contains a solver for the steady state and time dependent problems.

## Quick Start

To get started, create the needed subdirectories and run one of the `tests/time_dep_nse_.py` files, e.g.

```
pip install sadptprj_riclyap_adi
cd tests
mkdir data
mkdir results
# export PYTHONPATH="$PYTHONPATH:path/to/repo/"  # add the repo to the path
# pip install dolfin_navier_scipy                # or install the module using pip
python3 time_dep_nse_expnonl.py
```

Then, to examine the results, launch
```
paraview results/vel_TH__timestep.pvd
```

## Test Cases and Examples

A selection:

 * `tests/mini_setup.py`: a minimal setup for a steady-state simulation
 * `tests/steadystate_schaefer-turek_2D-1.py`: the 2D steady-state cylinder wake benchmark by *Sch&auml;fer/Turek*
 * `tests/steadystate_rotcyl.py`: the 2D cylinder wake with a freely rotating cylinder as benchmarked in *Richter et al.*
 * `tests/time_dep_nse_.py`: time integration with Picard and Newton linearization
 * `tests/time_dep_nse_expnonl.py`: time integration with explicit treatment of the nonlinearity
 * `tests/time_dep_nse_bcrob.py`: time integration of the cylinder wake with boundary controls
 * `tests/time_dep_nse_krylov.py`: time integration with iterative solves of the state equations via [`krypy`](https://github.com/andrenarchy/krypy)
 * `tests/time_dep_nse_double_rotcyl_bcrob.py`: rotating double cylinder via
   Robin boundary conditions

## Dependencies

 * dolfin interface to [FEniCS](https://fenicsproject.org/) -- tested with `v2019.2.0`, `v2018.1.0`, `v2017.2` 
 * [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi)

The latter is my home-brew module that includes the submodule `lin_alg_utils` with routines for solving the saddle point problem as it arises in the `(v,p)` formulation of the NSE. 

**Note**: the branch `lau-included` already contains the module `sadptprj_riclyap_adi`

## Documentation

Documentation of the code goes [here](http://dolfin-navier-scipy.readthedocs.org/en/latest/index.html).

## Installation as Module

```
pip install dolfin_navier_scipy
```
