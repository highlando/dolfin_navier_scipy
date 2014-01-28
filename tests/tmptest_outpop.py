# this is rather optical checking
import dolfin
import scipy.sparse.linalg as spsla
import numpy as np

import dolfin_to_nparrays as dtn
import cont_obs_utils as cou
import lin_alg_utils as lau

from optcont_main import drivcav_fems
dolfin.parameters.linear_algebra_backend = "uBLAS"

NV = 20
NY = 8

mesh = dolfin.UnitSquareMesh(NV, NV)
V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
Q = dolfin.FunctionSpace(mesh, "CG", 1)

dolfin.plot(mesh)

testcase = 3  # 2,3
# testvelocities
if testcase == 1:
    """case 1 -- not div free"""
    exv = dolfin.Expression(('x[1]', 'x[1]'))
if testcase == 2:
    """case 2 -- disc div free"""
    exv = dolfin.Expression(('1', '1'))
if testcase == 3:
    """case 3 -- cont div free"""
    import sympy as smp
    x, y = smp.symbols('x[0], x[1]')
    u_x = x*x*(1-x)*(1-x)*2*y*(1-y)*(2*y-1)
    u_y = y*y*(1-y)*(1-y)*2*x*(1-x)*(1-2*x)
    from sympy.printing import ccode
    exv = dolfin.Expression((ccode(u_x), ccode(u_y)))

testv = dolfin.interpolate(exv, V)

odcoo = dict(xmin=0.45,
             xmax=0.55,
             ymin=0.6,
             ymax=0.8)

# get the system matrices
femp = drivcav_fems(NV)
stokesmats = dtn.get_stokessysmats(femp['V'], femp['Q'], nu=1)
# remove the freedom in the pressure
stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]

bc = dolfin.DirichletBC(V, exv, 'on_boundary')

# reduce the matrices by resolving the BCs
(stokesmatsc,
 rhsd_stbc,
 invinds,
 bcinds,
 bcvals) = dtn.condense_sysmatsbybcs(stokesmats, [bc])

# check the C
MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY, NV=NV)
# MyC = MyC[:, invinds][:, :]


# signal space
ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])
Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

y1 = dolfin.Function(Y)
y2 = dolfin.Function(Y)
y3 = dolfin.Function(Y)

ptmct = lau.app_prj_via_sadpnt(amat=stokesmats['M'],
                               jmat=stokesmats['J'],
                               rhsv=MyC.T,
                               transposedprj=True)

testvi = testv.vector().array()
testvi0 = np.atleast_2d(testv.vector().array()).T
testvi0 = lau.app_prj_via_sadpnt(amat=stokesmats['M'],
                                 jmat=stokesmats['J'],
                                 rhsv=testvi0)


print "||J*v|| = {0}".format(np.linalg.norm(stokesmats['J'] * testvi))
print "||J* v_df|| = {0}".format(np.linalg.norm(stokesmats['J'] * testvi0))

# # testsignals from the test velocities
testy = spsla.spsolve(My, MyC * testvi)
testyv0 = spsla.spsolve(My, MyC * testvi0)
# testyg = spsla.spsolve(My, MyC * (testvi.flatten() - testvi0))
testry = spsla.spsolve(My, np.dot(ptmct.T, testvi))

print "||C v_df - C_df v|| = {0}".format(np.linalg.norm(testyv0 - testry))

y1.vector().set_local(testy[:NY])
dolfin.plot(y1, title="x-comp of C*v")

y2.vector().set_local(testy[NY:])
dolfin.plot(y2, title="y-comp of C*v")

# y2.vector().set_local(testyv0[:NY])
# dolfin.plot(y2, title="x-comp of $C*(P_{df}v)$")

# y3.vector().set_local(testyg[:NY])
# dolfin.plot(y3, title="x-comp of $C*(v - P_{df}v)$")

dolfin.interactive(True)
