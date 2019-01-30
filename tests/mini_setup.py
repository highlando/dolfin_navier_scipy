import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu
import numpy as np

N, Re, scheme, ppin = 2, 50, 'TH', None

femp, stokesmatsc, rhsd = \
    dnsps.get_sysmats(problem='cylinderwake', Re=Re,
                      scheme=scheme, mergerhs=True,
                      meshparams=dict(refinement_level=N))

Mc, Ac = stokesmatsc['M'], stokesmatsc['A']
BTc, Bc = stokesmatsc['JT'], stokesmatsc['J']
print(Bc.shape)

invinds = femp['invinds']

fv, fp = rhsd['fv'], rhsd['fp']
inivdict = dict(A=Ac, J=Bc, JT=BTc, M=Mc, ppin=ppin, fv=fv, fp=fp,
                return_vp=True, V=femp['V'],
                invinds=invinds, diribcs=femp['diribcs'])
vp_init = snu.solve_steadystate_nse(**inivdict)

NV = Bc.shape[1]

pfv = snu.get_pfromv(v=vp_init[0],
                     V=femp['V'], M=Mc, A=Ac, J=Bc, fv=fv,
                     invinds=femp['invinds'], diribcs=femp['diribcs'])
print(invinds.shape)
print(Bc.shape)
print(pfv.shape)
print(vp_init[0].shape)
print(vp_init[1].shape)

print(np.linalg.norm(pfv - vp_init[1]))
