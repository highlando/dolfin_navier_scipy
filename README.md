Documentation goes [here](http://dolfin-navier-scipy.readthedocs.org/en/latest/index.html).

To get started, create the needed subdirectories and run one of the `time_dep_nse_.py` files, e.g.

```
mkdir data
mkdir results
python time_dep_nse_.py
```

Then, to examine the results, launch
```
paraview results/vel_TH___timestep.pvd
```
