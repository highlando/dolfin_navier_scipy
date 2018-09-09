import dolfin
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import numpy as np
# from dolfin import dx, grad, inner

import dolfin_navier_scipy as dns
import dolfin_navier_scipy.dolfin_to_sparrays as dnsts
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou

# dolfin.parameters.linear_algebra_backend = "uBLAS"

linnsesol = True  # whether to linearize about the NSE solution
debug = False
# debug = True

timeint = False
compevs = False

# timeint = True
compevs = True


def linearzd_quadterm(H, linv, hlstr=None):
    print('TODO: this function will be deprecated soon')
    print('see ~/work/code/nse-quad-refree/python/conv_tensor_utils.py')
    print('for a maintained version')
    try:
        HLm = dou.load_spa(hlstr + '.mtx')
        print('loaded `hlmat`')
    except IOError:
        print('assembling hlmat ...')
        nv = linv.size
        # HLm = np.array(H * (sps.kron(sps.eye(nv), linv) +
        #                     sps.kron(linv, sps.eye(nv))))
        # that seems a fast option but too memory consuming for my laptop
        HL = []
        for k in range(nv):
            ek = np.zeros((nv, 1))
            ek[k] = 1
            H1k = sps.csr_matrix(H*np.kron(ek, linv))
            H2k = sps.csr_matrix(H*np.kron(linv, ek))
            HL.append(H1k + H2k)
        HLm = sps.hstack(HL)
        assert np.linalg.norm(2*H*np.kron(linv, linv) - HLm*linv) < 1e-12
        dou.save_spa(HLm, hlstr)
    return HLm


def test_qbdae_ass(problemname='cylinderwake', N=1, Re=None, nu=3e-2,
                   t0=0.0, tE=1.0, Nts=100, use_saved_mats=None):

    trange = np.linspace(t0, tE, Nts+1)
    DT = (tE-t0)/Nts
    rdir = 'results/'
    ddir = 'data/'

    if use_saved_mats is None:
        femp, stokesmatsc, rhsd_vfrc, rhsd_stbc = \
            dns.problem_setups.get_sysmats(problem=problemname, N=N, Re=Re)
        invinds = femp['invinds']

        A, J, M = stokesmatsc['A'], stokesmatsc['J'], stokesmatsc['M']
        L = 0*A
        fvc, fpc = rhsd_vfrc['fvc'], rhsd_vfrc['fpr']
        fv_stbc, fp_stbc = rhsd_stbc['fv'], rhsd_stbc['fp']

        hstr = ddir + problemname + '_N{0}_hmat'.format(N)
        try:
            hmat = dou.load_spa(hstr)
            print('loaded `hmat`')
        except IOError:
            print('assembling hmat ...')
            hmat = dnsts.ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)
            dou.save_spa(hmat, hstr)

        invinds = femp['invinds']
        NV, NP = invinds.shape[0], J.shape[0]
        zerv = np.zeros((NV, 1))

        bc_conv, bc_rhs_conv, rhsbc_convbc = \
            snu.get_v_conv_conts(prev_v=zerv, V=femp['V'], invinds=invinds,
                                 diribcs=femp['diribcs'], Picard=False)
        fp = fp_stbc + fpc
        fv = fv_stbc + fvc - bc_rhs_conv

        if linnsesol:
            vp_nse, _ = snu.\
                solve_steadystate_nse(A=A, J=J, JT=None, M=M,
                                      fv=fv_stbc + fvc,
                                      fp=fp_stbc + fpc,
                                      V=femp['V'], Q=femp['Q'],
                                      invinds=invinds,
                                      diribcs=femp['diribcs'],
                                      return_vp=False, ppin=-1,
                                      N=N, nu=nu,
                                      clearprvdata=False)
            old_v = vp_nse[:NV]
        else:
            import sadptprj_riclyap_adi.lin_alg_utils as lau
            # Stokes solution as initial value
            vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J,
                                             rhsv=fv_stbc + fvc,
                                             rhsp=fp_stbc + fpc)

            old_v = vp_stokes[:NV]

        sysmat = sps.vstack([sps.hstack([M+DT*(A+bc_conv), J.T]),
                             sps.hstack([J, sps.csc_matrix((NP, NP))])])

    if use_saved_mats is not None:
        # if saved as in ../get_exp_mats
        import scipy.io
        mats = scipy.io.loadmat(use_saved_mats)
        A = - mats['A']
        L = - mats['L']
        Re = mats['Re']
        N = A.shape[0]
        M = mats['M']
        J = mats['J']
        hmat = -mats['H']
        fv = mats['fv']
        fp = mats['fp']
        NV, NP = fv.shape[0], fp.shape[0]
        old_v = mats['ss_stokes']
        sysmat = sps.vstack([sps.hstack([M+DT*A, J.T]),
                             sps.hstack([J, sps.csc_matrix((NP, NP))])])

    if compevs:
        import matplotlib.pyplot as plt
        import scipy.linalg as spla

        hlstr = ddir + problemname + '_N{0}_Re{1}Nse{2}_hlmat'.\
            format(N, Re, linnsesol)
        HL = linearzd_quadterm(hmat, old_v, hlstr=hlstr)
        print(HL.shape)
        asysmat = sps.vstack([sps.hstack([-(A-L+HL), J.T]),
                              sps.hstack([J, sps.csc_matrix((NP, NP))])])
        msysmat = sps.vstack([sps.hstack([M, sps.csc_matrix((NV, NP))]),
                              sps.hstack([sps.csc_matrix((NP, NV)),
                                          sps.csc_matrix((NP, NP))])])
        levstr = ddir + problemname + '_N{0}Re{1}Nse{2}_levs'.\
            format(N, Re, linnsesol)
        try:
            levs = dou.load_npa(levstr)
            if debug:
                raise IOError()
            print('loaded the eigenvalues of the linearized system')
        except IOError:
            print('computing the eigenvalues of the linearized system')
            A = asysmat.todense()
            M = msysmat.todense()
            levs = spla.eigvals(A, M, overwrite_a=True, check_finite=False)
            dou.save_npa(levs, levstr)

        plt.figure(1)
        # plt.xlim((-25, 15))
        # plt.ylim((-50, 50))
        plt.plot(np.real(levs), np.imag(levs), '+')
        plt.show(block=False)

    if timeint:
        print('computing LU once...')
        sysmati = spsla.factorized(sysmat)

        vfile = dolfin.File(rdir + problemname + 'qdae__vel.pvd')
        pfile = dolfin.File(rdir + problemname + 'qdae__p.pvd')

        prvoutdict = dict(V=femp['V'], Q=femp['Q'], vfile=vfile, pfile=pfile,
                          invinds=invinds, diribcs=femp['diribcs'],
                          vp=None, t=None, writeoutput=True)

        print('doing the time loop...')
        for t in trange:
            crhsv = M*old_v + DT*(fv - hmat*np.kron(old_v, old_v))
            crhs = np.vstack([crhsv, fp])
            vp_new = np.atleast_2d(sysmati(crhs.flatten())).T

            prvoutdict.update(dict(vp=vp_new, t=t))
            dou.output_paraview(**prvoutdict)

            old_v = vp_new[:NV]
            print(t, np.linalg.norm(old_v))

if __name__ == '__main__':
    # test_qbdae_ass(problemname='cylinderwake', N=1, Re=1e2, tE=2.0, Nts=800)
    test_qbdae_ass(problemname='cylinderwake', tE=2.0, Nts=800,
                   use_saved_mats='../data/' +
                   # 'cylinderwakequadform__mats_N3022_Re100.mat')
                   'cylinderwakequadform__mats_N5812_Re100.mat')
