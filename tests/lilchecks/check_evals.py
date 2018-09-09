import scipy.io
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import dolfin_navier_scipy.data_output_utils as dou
import matplotlib.pyplot as plt
import scipy.linalg as spla
import numpy as np

from solve_nse_quadraticterm import linearzd_quadterm

Re = 150
NN = 3022  # 9356  # 5812  # 3022
ddir = 'data/'
problemname = 'cylinderwake'

debug = False
compall = True
# compall = False
shiftall = True

linsysstr = '../data/cylinderwake__mats_N{0}_Re{1}.mat'.format(NN, Re)
quadsysstr = '../data/cylinderwakequadform__mats_N{0}_Re{1}.mat'.format(NN, Re)

linsysmats = scipy.io.loadmat(linsysstr)
quadsysmats = scipy.io.loadmat(quadsysstr)

hlstr = ddir + problemname + '_N{0}_Re{1}_hlmat'.format(NN, Re)

hmat = quadsysmats['H']
Lquad = quadsysmats['L']
Aquad = quadsysmats['A']
vssnse = quadsysmats['v_ss_nse']
HL = linearzd_quadterm(hmat, vssnse, hlstr=hlstr)

A = linsysmats['A']
M = linsysmats['M']
J = linsysmats['J']
NV, NP = A.shape[0], J.shape[0]

tstvec = np.random.randn(NV, 1)
print np.linalg.norm(A*vssnse - Aquad*vssnse - HL*vssnse - 0*Lquad*vssnse)
print np.linalg.norm(A*tstvec - Aquad*tstvec - HL*tstvec)

# raise Warning('TODO: debug')

shift = 0.3e-1

asysmat = sps.vstack([sps.hstack([A, J.T]),
                      sps.hstack([J, sps.csc_matrix((NP, NP))])])
msysmat = sps.vstack([sps.hstack([M, sps.csc_matrix((NV, NP))]),
                      sps.csc_matrix((NP, NV+NP))])
mshiftsmat = sps.vstack([sps.hstack([M, shift*J.T]),
                         sps.hstack([shift*J, sps.csc_matrix((NP, NP))])])
levstr = ddir + problemname + '_N{0}Re{1}linsys_levs_compall'.\
    format(NN, Re, compall)
try:
    if debug:
        raise IOError()
    levs = dou.load_npa(levstr)
    print 'loaded the eigenvalues of the linearized system'
except IOError:
    print 'computing the eigenvalues of the linearized system'
    if compall:
        A = asysmat.todense()
        M = msysmat.todense() if not shiftall else mshiftsmat.todense()
        levs = spla.eigvals(A, M, overwrite_a=True, check_finite=False)
    else:
        levs = spsla.eigs(asysmat, M=mshiftsmat, sigma=1, k=10, which='LR',
                          return_eigenvectors=False)
    dou.save_npa(levs, levstr)

plt.figure(1)
# plt.xlim((-20, 15))
# plt.ylim((-100, 100))
plt.plot(np.real(levs), np.imag(levs), '+')
plt.show(block=False)

projsys = False
if projsys:
    mfac = spsla.splu(M)
    mijtl = []
    for kcol in range(J.shape[0]):
        curcol = np.array(J.T[:, kcol].todense())
        micc = np.array(mfac.solve(curcol.flatten()))
        mijtl.append(micc.reshape((NV, 1)))

    mijt = np.hstack(mijtl)
    si = np.linalg.inv(J*mijt)

    def _comp_proj(v):
        jtsijv = np.array(J.T*np.dot(si, J*v)).flatten()
        mjtsijv = mfac.solve(jtsijv)
        return v - mjtsijv.reshape((NV, 1))

    def _comp_proj_t(v):
        jtsijv = np.array(J.T*np.dot(si, J*v)).flatten()
        mjtsijv = mfac.solve(jtsijv)
        return v - mjtsijv.reshape((NV, 1))

    v = np.random.randn(A.shape[0], 1)
    pv = _comp_proj(v)
    raise Warning('TODO: debug')
