Documentation of the code goes [here](http://dolfin-navier-scipy.readthedocs.org/en/latest/index.html).


To get started, create the needed subdirectories and run one of the `tests/time_dep_nse_.py` files, e.g.

```
cd tests
mkdir data
mkdir results
# export PYTHONPATH="$PYTHONPATH:path/to/repo/"  # add the repo to the path
python3 time_dep_nse_expnonl.py
```

Then, to examine the results, launch
```
paraview results/vel_TH__timestep.pvd
```

The test cases (a selection):

 * `tests/mini_setup.py`: a minimal setup for a steady-state simulation
 * `tests/steadystate_schaefer-turek_2D-1.py`: the 2D steady-state cylinder wake benchmark by *Sch&auml;fer/Turek*
 * `tests/steadystate_rotcyl.py`: the 2D cylinder wake with a freely rotating cylinder as benchmarked in *Richter et al.*
 * `tests/time_dep_nse_.py`: time integration with Picard and Newton linearization
 * `tests/time_dep_nse_expnonl.py`: time integration with explicit treatment of the nonlinearity
 * `tests/time_dep_nse_bcrob.py`: time integration of the cylinder wake with boundary controls
 * `tests/time_dep_nse_krylov.py`: time integration with iterative solves of the state equations via [`krypy`](hellooo)

Dependencies
---

 * dolfin interface to [FEniCS](https://fenicsproject.org/) -- tested with `v2018.1.0`, `v2017.2` 
 * [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi)

The latter is my home-brew module that includes the submodule `lin_alg_utils` with routines for solving the saddle point problem as it arises in the `(v,p)` formulation of the NSE. 

**Note**: the branch `gh-deps-included` already contains the module `sadptprj_riclyap_adi`
