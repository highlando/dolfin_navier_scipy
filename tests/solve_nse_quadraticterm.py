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
                   t0=0.0, tE=1.0, Nts=100):

    trange = np.linspace(t0, tE, Nts+1)
    DT = (tE-t0)/Nts
    print DT
    rdir = 'results/'

    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc, \
        data_prfx, ddir, proutdir = \
        dns.problem_setups.get_sysmats(problem=problemname, N=N, Re=Re)

    invinds = femp['invinds']
    A, J, M = stokesmatsc['A'], stokesmatsc['J'], stokesmatsc['M']
    fvc, fpc = rhsd_vfrc['fvc'], rhsd_vfrc['fpr']
    fv_stbc, fp_stbc = rhsd_stbc['fv'], rhsd_stbc['fp']

    print 'assembling hmat ...'
    hmat = dnsts.ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)
    # print 'maybe not'

    invinds = femp['invinds']
    NV, NP = invinds.shape[0], J.shape[0]
    zerv = np.zeros((NV, 1))

    bc_conv, bc_rhs_conv, rhsbc_convbc = \
        snu.get_v_conv_conts(prev_v=zerv, V=femp['V'], invinds=invinds,
                             diribcs=femp['diribcs'], Picard=False)

    # Stokes solution as initial value
    vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J,
                                     rhsv=fv_stbc + fvc,
                                     rhsp=fp_stbc + fpc)
    old_v = vp_stokes[:NV]

    sysmat = sps.vstack([sps.hstack([M+DT*A, J.T]),
                         sps.hstack([J, sps.csc_matrix((NP, NP))])])

    print 'computing LU once...'
    sysmati = spsla.factorized(sysmat)

    vfile = dolfin.File(rdir + problemname + 'qdae__vel.pvd')
    pfile = dolfin.File(rdir + problemname + 'qdae__p.pvd')

    # snu.solve_nse(A=A, M=M, J=J, JT=None,
    #               fvc=fvc, fpr=fpc,
    #               fv_stbc=fv_stbc, fp_stbc=fp_stbc,
    #               trange=trange,
    #               t0=None, tE=None, Nts=None,
    #               V=femp['V'], Q=femp['Q'],
    #               invinds=invinds, diribcs=femp['diribcs'],
    #               N=N, nu=femp['nu'],
    #               vel_nwtn_stps=1, vel_nwtn_tol=5e-15,
    #               get_datastring=None,
    #               data_prfx='',
    #               paraviewoutput=True, prfdir='',
    #               vfileprfx=rdir + problemname + 'qdae__vel',
    #               pfileprfx=rdir + problemname + 'qdae__p',
    #               return_dictofvelstrs=False,
    #               comp_nonl_semexp=True)
    # raise Warning('TODO: debug')

    prvoutdict = dict(V=femp['V'], Q=femp['Q'], vfile=vfile, pfile=pfile,
                      invinds=invinds, diribcs=femp['diribcs'],
                      vp=None, t=None, writeoutput=True)

    print 'doing the time loop...'
    for t in trange:
        conv_mat, rhs_conv, rhsbc_conv = \
            snu.get_v_conv_conts(prev_v=old_v, V=femp['V'], invinds=invinds,
                                 diribcs=femp['diribcs'], Picard=False)
        # crhsv = M*old_v + DT*(fv_stbc + fvc + rhs_conv + rhsbc_conv
        #                       - conv_mat*old_v)
        # raise Warning('TODO: debug')
        crhsv = M*old_v + DT*(fv_stbc + fvc + bc_rhs_conv + rhsbc_convbc
                              - bc_conv*old_v - hmat*np.kron(old_v, old_v))
        crhs = np.vstack([crhsv, fp_stbc + fpc])
        vp_new = np.atleast_2d(sysmati(crhs.flatten())).T
        # vp_new = lau.solve_sadpnt_smw(amat=M+DT*(A+0*conv_mat), jmat=J,
        #                               rhsv=crhsv,
        #                               rhsp=fp_stbc + fpc)

        prvoutdict.update(dict(vp=vp_new, t=t))
        dou.output_paraview(**prvoutdict)

        old_v = vp_new[:NV]
        print t, np.linalg.norm(old_v)

if __name__ == '__main__':
    test_qbdae_ass(problemname='cylinderwake', N=2, Re=1e2, tE=1.0, Nts=500)
