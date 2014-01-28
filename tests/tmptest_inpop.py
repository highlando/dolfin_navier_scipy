# this is rather optical checking
import dolfin
import numpy as np
import scipy.sparse.linalg as spsla

import dolfin_to_nparrays as dtn
import cont_obs_utils as cou

dolfin.parameters.linear_algebra_backend = "uBLAS"

NV = 20
mesh = dolfin.UnitSquareMesh(NV, NV)
V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
Q = dolfin.FunctionSpace(mesh, "CG", 1)

NU = 5

cdcoo = dict(xmin=0.4,
             xmax=0.6,
             ymin=0.2,
             ymax=0.3)

cdom = cou.ContDomain(cdcoo)

# get the system matrices
stokesmats = dtn.get_stokessysmats(V, Q)

# check the B
B, Mu = cou.get_inp_opa(cdcoo=cdcoo, V=V, NU=NU)

# get the rhs expression of Bu
Bu = spsla.spsolve(stokesmats['M'], B*np.vstack([1*np.ones((NU, 1)),
                                                 1*np.ones((NU, 1))]))

bu = dolfin.Function(V)
bu.vector().set_local(Bu)
bu.vector()[2] = 1
dolfin.plot(bu, title='plot of Bu')

dolfin.interactive(True)
