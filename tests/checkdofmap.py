import numpy as np
import dolfin
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

N = 3
femp = dnsps.drivcav_fems(N)

stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'])

# reduce the matrices by resolving the BCs
(stokesmatsc,
 rhsd_stbc,
 invinds,
 bcinds,
 bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                     femp['diribcs'])

fv = dolfin.Constant(('1', '0'))
v = dolfin.interpolate(fv, femp['V'])

invals = np.zeros(invinds.size)

coorar, xinds, yinds = dts.get_dof_coors(femp['V'])
icoorar, ixinds, iyinds = dts.get_dof_coors(femp['V'], invinds=invinds)

invals[iyinds] = 1

print coorar, xinds
print icoorar, ixinds

print v.vector().array()

v.vector()[invinds] += invals
print v.vector().array()[xinds]
print v.vector().array()[yinds]
dolfin.plot(v)
