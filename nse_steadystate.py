import dolfin
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import numpy as np

import scipy.io


def solve_steady_nse(pathtomats=None, nnwtnstps=3, npcrdstps=1):
    mats = scipy.io.loadmat(pathtomats) 
    A = mats['A']
    H = mats['H']
    L1, L2 = mats['L1'], mats['L2']
    J = mats['J']
    fv = mats['fv']
    fv_diff = mats['fv_diff']
    fv_conv = mats['fv_conv']
    fp = mats['fp']
    fp_div = mats['fp_div']

    NV, NP = fv.shape[0], fp.shape[0]

    # Stokes solution to start the iteration
    stmat = sps.vstack([sps.hstack([A, J.T]),
                        sps.hstack([J, sps.csc_matrix((NP, NP))])])
    strhs = np.vstack([fv - fv_diff, fp - fp_div])
    vpst = sps.spsolve(stmat, strhs)
    vst = (vpst[:NV]).reshape((NV, 1))


def comp_linh(H, linv, lintype='Newton'):
    """ compute the linearization of `H*kron(., .)` about linv`, namely

    `Hl(.) := H*(linv, .)` for Picard scheme or
    `Hl(.) := H*(linv, .) + H*(., linv)` for Newton scheme
    """

    listofcols = []
    Nv = linv.size()
    for curcol in range(Nv):
        pass



if __name__ == '__main__':
    print 'blablabla'
