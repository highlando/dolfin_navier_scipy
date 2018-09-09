Documentation goes [here](http://dolfin-navier-scipy.readthedocs.org/en/latest/index.html).


To get started, create the needed subdirectories and run one of the `tests/time_dep_nse_.py` files, e.g.

```
cd tests
mkdir data
mkdir results
python3 time_dep_nse_.py
```

Then, to examine the results, launch
```
paraview results/vel_TH___timestep.pvd
```

Dependencies
---

Note: The branch `lau-included` already includes my homebrew submodule. 

 * dolfin interface to [FEniCS](https://fenicsproject.org/) (v2018.1.0)
 * [sadptprj_riclyap_adi](https://github.com/highlando/sadptprj_riclyap_adi)

The latter is my home-brew module that includes the submodule `lin_alg_utils` with routines for solving the saddle point problem as it arises in the `(v,p)` formulation of the NSE. 

