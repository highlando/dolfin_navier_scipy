import dolfin
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import numpy as np
# from dolfin import dx, grad, inner

import dolfin_navier_scipy as dns
import dolfin_navier_scipy.dolfin_to_sparrays as dnsts
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau
# dolfin.parameters.linear_algebra_backend = "uBLAS"


def test_qbdae_ass(problemname='cylinderwake', N=1, Re=4e2, nu=3e-2,
                   t0=0.0, tE=1.0, Nts=100, use_saved_mats=None):

    trange = np.linspace(t0, tE, Nts+1)
    DT = (tE-t0)/Nts
    rdir = 'results/'
    ddir = 'data/'
    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc = \
        dns.problem_setups.get_sysmats(problem=problemname, N=N, Re=Re)
    invinds = femp['invinds']

    if use_saved_mats is None:

        A, J, M = stokesmatsc['A'], stokesmatsc['J'], stokesmatsc['M']
        fvc, fpc = rhsd_vfrc['fvc'], rhsd_vfrc['fpr']
        fv_stbc, fp_stbc = rhsd_stbc['fv'], rhsd_stbc['fp']

        hstr = ddir + problemname + '_N{0}_hmat'.format(N)
        try:
            hmat = dou.load_spa(hstr)
            print 'loaded `hmat`'
        except IOError:
            print 'assembling hmat ...'
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
        M = mats['M']
        J = mats['J']
        hmat = -mats['H']
        fv = mats['fv']
        fp = mats['fp']
        NV, NP = fv.shape[0], fp.shape[0]
        old_v = mats['ss_stokes']
        sysmat = sps.vstack([sps.hstack([M+DT*A, J.T]),
                             sps.hstack([J, sps.csc_matrix((NP, NP))])])

    print 'computing LU once...'
    sysmati = spsla.factorized(sysmat)

    vfile = dolfin.File(rdir + problemname + 'qdae__vel.pvd')
    pfile = dolfin.File(rdir + problemname + 'qdae__p.pvd')

    prvoutdict = dict(V=femp['V'], Q=femp['Q'], vfile=vfile, pfile=pfile,
                      invinds=invinds, diribcs=femp['diribcs'],
                      vp=None, t=None, writeoutput=True)

    print 'doing the time loop...'
    for t in trange:
        # conv_mat, rhs_conv, rhsbc_conv = \
        #     snu.get_v_conv_conts(prev_v=old_v, V=femp['V'], invinds=invinds,
        #                          diribcs=femp['diribcs'], Picard=False)
        # crhsv = M*old_v + DT*(fv_stbc + fvc + rhs_conv + rhsbc_conv
        #                       - conv_mat*old_v)
        crhsv = M*old_v + DT*(fv - hmat*np.kron(old_v, old_v))
        crhs = np.vstack([crhsv, fp])
        vp_new = np.atleast_2d(sysmati(crhs.flatten())).T
        # vp_new = lau.solve_sadpnt_smw(amat=M+DT*(A+0*conv_mat), jmat=J,
        #                               rhsv=crhsv,
        #                               rhsp=fp_stbc + fpc)

        prvoutdict.update(dict(vp=vp_new, t=t))
        dou.output_paraview(**prvoutdict)

        old_v = vp_new[:NV]
        print t, np.linalg.norm(old_v)

if __name__ == '__main__':
    test_qbdae_ass(problemname='cylinderwake', N=2, Re=1e2, tE=2.0, Nts=800)
    # test_qbdae_ass(problemname='cylinderwake', N=1, Re=1e2, tE=2.0, Nts=800,
    #                use_saved_mats='../data/' +
    #                'cylinderwakequadform__mats_N5812_Re100.0.mat')
