import numpy as np
import dolfin
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

N = 2
femp = dnsps.drivcav_fems(N)
mesh = dolfin.UnitSquareMesh(N, N)

stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'])

# reduce the matrices by resolving the BCs
(stokesmatsc,
 rhsd_stbc,
 invinds,
 bcinds,
 bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                     femp['diribcs'])

fv = dolfin.Constant(('0', '1'))
v = dolfin.interpolate(fv, femp['V'])

invals = np.zeros(invinds.size)

coorar, xinds, yinds, corvec = dts.get_dof_coors(femp['V'])
icoorar, ixinds, iyinds, icorvec = dts.get_dof_coors(femp['V'],
                                                     invinds=invinds)

invals[ixinds] = icoorar[:, 0]

# print coorar, xinds
# print icoorar, ixinds

# print v.vector().array()

print v.vector().array()[invinds]
v.vector()[invinds] += invals
# print v.vector().array()[xinds]
# print v.vector().array()[yinds]
print v.vector().array()[invinds]
print icoorar[:, 1], iyinds
print icorvec
dolfin.plot(v)
dolfin.plot(mesh)
dolfin.interactive()
