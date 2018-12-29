import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu
import numpy as np

N, Re, scheme, ppin = 2, 50, 'TH', None

femp, stokesmatsc, rhsd = \
    dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                      scheme=scheme, mergerhs=True)

Mc, Ac = stokesmatsc['M'], stokesmatsc['A']
BTc, Bc = stokesmatsc['JT'], stokesmatsc['J']
invinds = femp['invinds']

fv, fp = rhsd['fv'], rhsd['fp']
inivdict = dict(A=Ac, J=Bc, JT=BTc, M=Mc, ppin=ppin, fv=fv, fp=fp,
                return_vp=True, V=femp['V'],
                invinds=invinds, diribcs=femp['diribcs'])

# ## Solve the steady-state NSE
vp_steadystate = snu.solve_steadystate_nse(**inivdict)

NV = Bc.shape[1]

# ## Test: recompute the p from the v
pfv = snu.get_pfromv(v=vp_steadystate[:NV, :],
                     V=femp['V'], M=Mc, A=Ac, J=Bc, fv=fv,
                     invinds=femp['invinds'], diribcs=femp['diribcs'])

print('Number of inner velocity nodes: {0}'.format(invinds.shape))
print('Shape of the divergence matrix: ', Bc.shape)

print('error in recomputed pressure: {0}'.
      format(np.linalg.norm(pfv - vp_steadystate[NV:, :])))
