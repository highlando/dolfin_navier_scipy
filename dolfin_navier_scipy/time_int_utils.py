import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import time
import logging

from rich.progress import track

import sadptprj_riclyap_adi.lin_alg_utils as lau

# from dolfin_navier_scipy.residual_checks import get_imex_res
# from dolfin_navier_scipy.dolfin_to_sparrays import expand_vp_dolfunc

__all__ = ['cnab',
           'sbdftwo',
           'nse_include_lnrcntrllr',
           'semi_implicit_euler'
           ]


def cnab(trange=None, inivel=None, inip=None, bcs_ini=[],
         M=None, A=None, J=None,
         f_vdp=None,
         f_tdp=None, g_tdp=None,
         f_tvdp=None,
         scalep=-1.,
         getbcs=None, applybcs=None, appndbcs=None,
         savevp=None, dynamic_rhs=None, dynamic_rhs_memory={},
         # check_ff=False,
         check_ff_maxv=None,
         # implicit_dynamic_rhs=None, implicit_dynamic_rhs_memory={},
         ntimeslices=10, verbose=True):
    """
    to be provided

    """

    dt, listofts = _inittimegrid(trange, ntimeslices=ntimeslices)

    NP, NV = J.shape
    zerorhs = np.zeros((NV, 1))
    ffflag = 0

    if dynamic_rhs is None:
        def dynamic_rhs(t, vc=None, memory={}, mode=None):
            return zerorhs, memory
    else:
        pass

    if f_tvdp is not None:
        def _dynamic_rhs(t, vc=None, memory={}, mode=None):
            _cfv, _mmry = dynamic_rhs(t, vc=vc, memory=memory, mode=mode)
            # logging.info('set zero for DEBUG')
            # logging.info(f't:{t} -- rhsval:{np.linalg.norm(f_tvdp(t, vc))}')
            return _cfv+f_tvdp(t, vc), _mmry
    else:
        _dynamic_rhs = dynamic_rhs

    if f_vdp is None:
        def f_vdp(vvec):
            return zerorhs

    dfv_c, drm = _dynamic_rhs(trange[0], vc=inivel,
                              memory=dynamic_rhs_memory, mode='init')

    # if implicit_dynamic_rhs is None:
    #     def implicit_dynamic_rhs(t, vc=None, memory={}, mode=None):
    #         return zerorhs, memory

    # mplctdnmc_rhs = implicit_dynamic_rhs  # rename for efficiency in typing
    # drm = dynamic_rhs_memory
    # mddrm = implicit_dynamic_rhs_memory

    savevp(appndbcs(inivel, bcs_ini), inip, time=trange[0])

    v_n, p_n, bcs_n, bfv_n, mbc_c, mbc_n, fv_n, nfc_c, nfc_n, dfv_n, drm \
        = _onestepheun(vc=inivel, pc=inip, tc=trange[0], tn=trange[1],
                       M=M, A=A, J=J,
                       scalep=scalep,
                       dfv_c=dfv_c, dynamic_rhs=_dynamic_rhs, drm=drm,
                       bcs_c=bcs_ini, applybcs=applybcs,
                       appndbcs=appndbcs, getbcs=getbcs,
                       f_tdp=f_tdp, f_vdp=f_vdp, g_tdp=g_tdp)

    savevp(appndbcs(v_n, bcs_n), p_n, time=trange[1])

    trpz_coeffmat = sps.vstack([sps.hstack([M+.5*dt*A, J.T]),
                                sps.hstack([J, sps.csr_matrix((NP, NP))])])
    coeffmatlu = spsla.factorized(trpz_coeffmat)

    for kck, ctrange in enumerate(listofts):
        nrmvc = np.linalg.norm(v_n)
        if verbose:
            logging.info(f'time {kck}/{ntimeslices} ' +
                         f'-- @runtime {time.process_time():.1f} ' +
                         f' -- |v| {nrmvc:.2e}')
        if nrmvc > check_ff_maxv or np.isnan(nrmvc):
            logging.warning('BREAK: |v| is `NaN` or ' +
                            f'|v| > threshhold({check_ff_maxv})')
            ffflag = 1
            break
        for ctime in ctrange:
            # bump the variables
            v_c, p_c = v_n, p_n
            bcs_c, bfv_c, mbc_c = bcs_n, bfv_n, mbc_n
            fv_c = fv_n
            dfv_c = dfv_n

            # the static parts
            nfc_o = nfc_c
            nfc_c = f_vdp(appndbcs(v_c, bcs_c))

            # predict the boundary conditions
            bcs_n = getbcs(ctime, appndbcs(v_c, bcs_c), p_c, mode='abtwo')
            bfv_n, bfp_n, mbc_n = applybcs(bcs_n)

            # new values of the explicit time parts
            fv_n, fp_n = f_tdp(ctime), g_tdp(ctime)

            dfv_n, drm = _dynamic_rhs(ctime, vc=v_c, memory=drm,
                                      mode='abtwo')

            rhs_n = M*v_c - .5*dt*A*v_c \
                - (mbc_n-mbc_c) \
                + .5*dt*(3*nfc_c-nfc_o) \
                + .5*dt*(fv_c+fv_n + bfv_n+bfv_c + dfv_n+dfv_c)
            # rhsnrm = np.linalg.norm(fv_c+fv_n + dfv_n+dfv_c)
            # cnvnrm = np.linalg.norm(3*nfc_c-nfc_o)
            # logging.info(f't:{ctime} -- rhsval:{rhsnrm}')
            # logging.info(f't:{ctime} -- cnvval:{cnvnrm}')

            vp_n = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())

            v_n = vp_n[:NV].reshape((NV, 1))
            p_n = 1./dt*scalep*vp_n[NV:].reshape((NP, 1))

            # updating the implicit rhs
            # mplct_dfv_o = mplct_dfv_c
            # mplct_dfv_c, mddrm = mplctdnmc_rhs(ctime, vc=v_new, memory=mddrm)

            savevp(appndbcs(v_n, bcs_n), p_n, time=ctime)

    return v_n, p_n, ffflag


def get_heunab_lti(hb=None, ha=None, hc=None, inihx=None, drift=None):
    """ realizes the Heun/AB2 discretization of a linear observer

    ```
    hx' = hA*hx + hb*y
     u  = hc*hx
    ```

    e.g. with `hb=C*C` and `hc=BB*X` to get output based feedback actuation
    """

    print('NOTE: HEUN+AB2 for the controller')

    def heunab_lti(t, vc=None, memory={}, mode='abtwo'):
        if mode == 'init':
            chx = inihx
            hcchx = hc.dot(chx)
            memory.update(dict(lastt=t, lasthx=inihx))
            return hcchx, memory

        if mode == 'heunpred' or mode == 'heuncorr':
            curdt = t - memory['lastt']
            if mode == 'heunpred':
                currhs = ha.dot(inihx) + hb.dot(vc) + drift(memory['lastt'])
                chx = inihx + curdt*currhs
                hcchx = hc.dot(chx)
                memory.update(dict(lastrhs=currhs))
                memory.update(dict(hphx=chx))
                return hcchx, memory

            elif mode == 'heuncorr':
                currhs = ha.dot(memory['hphx']) + hb.dot(vc) + drift(t)
                chx = inihx + .5*curdt*(currhs + memory['lastrhs'])
                hcchx = hc.dot(chx)
                memory.update(dict(lastt=t, lasthx=chx, lastdt=curdt))
                return hcchx, memory

        elif mode == 'abtwo':
            curdt = t - memory['lastt']
            currhs = ha.dot(memory['lasthx']) + hb.dot(vc) \
                + drift(memory['lastt'])
            chx = memory['lasthx'] + 1.5*curdt*currhs \
                - .5*memory['lastdt']*memory['lastrhs']
            memory.update(dict(lastt=t, lasthx=chx, lastrhs=currhs,
                               lastdt=curdt))
            hcchx = hc.dot(chx)
            return hcchx, memory

    return heunab_lti


def get_heuntrpz_lti(hb=None, ha=None, hc=None, inihx=None, drift=None,
                     constdt=None):
    """ realizes the Heun/trapezoidal discretization of a linear observer

    ```
    hx' = hA*hx + hb*y
     u  = hc*hx
    ```

    e.g. with `hb=C*C` and `hc=BB*X` to get output based feedback actuation
    """
    print('NOTE: HEUN+Implicit Trapezoidal rule for the controller')
    hN = ha.shape[0]
    cdt = constdt
    if constdt is not None:
        obsitmat = np.linalg.inv(np.eye(hN)-constdt/2*ha)
        print('NOTE: uniform time grid is assumed for the controller')
    else:
        raise NotImplementedError()

    def heuntrpz_lti(t, vc=None, memory={}, mode='abtwo'):
        if mode == 'init':
            chx = inihx
            hcchx = hc.dot(chx)
            memory.update(dict(lastt=t, lasthx=inihx))
            return hcchx, memory

        if mode == 'heunpred' or mode == 'heuncorr':
            if mode == 'heunpred':
                currhs = hb.dot(vc) + drift(t)
                chx = inihx + cdt*(ha@inihx + currhs)
                hcchx = hc.dot(chx)
                memory.update(dict(lastrhs=currhs, lasthx=inihx))
                memory.update(dict(hphx=chx))
                return hcchx, memory

            elif mode == 'heuncorr':
                currhs = hb.dot(vc) + drift(t)
                hphx = memory['hphx']
                lhx = memory['lasthx']
                lrhs = memory['lastrhs']
                chx = inihx + .5*cdt*(ha@(hphx+lhx) + currhs+lrhs)
                hcchx = hc.dot(chx)
                memory.update(dict(lastt=t, hchx=chx))
                return hcchx, memory

        else:
            # curdt = t - memory['lastt']
            crhs = hb.dot(vc) + drift(t)
            lrhs = memory['lastrhs']
            lhx = memory['lasthx']
            # <-- implicit trap rule
            chx = obsitmat@(lhx + .5*cdt*(ha@lhx + crhs + lrhs))
            # implicit trap rule -->
            memory.update(dict(lasthx=chx, lastrhs=crhs))
            hcchx = hc.dot(chx)
            return hcchx, memory

    return heuntrpz_lti


def sbdftwo(trange=None, inivel=None, inip=None, bcs_ini=[],
            M=None, A=None, J=None,
            f_vdp=None, f_tdp=None, g_tdp=None,
            check_ff=False, check_ff_maxv=None,
            scalep=-1.,
            getbcs=None, applybcs=None, appndbcs=None,
            savevp=None, dynamic_rhs=None, dynamic_rhs_memory={},
            # implicit_dynamic_rhs=None, implicit_dynamic_rhs_memory={},
            ntimeslices=10, verbose=True):
    """
    to be provided

    """

    dt, listofts = _inittimegrid(trange, ntimeslices=ntimeslices)

    NP, NV = J.shape
    zerorhs = np.zeros((NV, 1))

    if dynamic_rhs is None:
        def dynamic_rhs(t, vc=None, memory={}, mode=None):
            return zerorhs, memory

    if f_vdp is None:
        def f_vdp(vvec):
            return zerorhs

    dfv_c, drm = dynamic_rhs(trange[0], vc=inivel,
                             memory=dynamic_rhs_memory, mode='init')

    savevp(appndbcs(inivel, bcs_ini), inip, time=trange[0])

    v_c = inivel
    v_n, p_n, bcs_n, bfv_n, mbc_c, mbc_n, fv_n, nfc_c, nfc_n, dfv_n, drm \
        = _onestepheun(vc=v_c, pc=inip, tc=trange[0], tn=trange[1],
                       M=M, A=A, J=J,
                       scalep=scalep,
                       dfv_c=dfv_c, dynamic_rhs=dynamic_rhs, drm=drm,
                       bcs_c=bcs_ini, applybcs=applybcs,
                       appndbcs=appndbcs, getbcs=getbcs,
                       f_tdp=f_tdp, f_vdp=f_vdp, g_tdp=g_tdp)

    savevp(appndbcs(v_n, bcs_n), p_n, time=trange[1])

    bdft_coeffmat = sps.vstack([sps.hstack([M+2./3*dt*A, J.T]),
                                sps.hstack([J, sps.csr_matrix((NP, NP))])])
    coeffmatlu = spsla.factorized(bdft_coeffmat)

    ffflag = 0

    for kck, ctrange in enumerate(listofts):
        nrmvc = np.linalg.norm(v_c)
        if verbose:
            print('time-stepping {0}/{1} complete -- @runtime {2:.1f} '.
                  format(kck, ntimeslices, time.process_time()) +
                  ' -- |v| {0:.2e}'.format(nrmvc))

        if nrmvc > check_ff_maxv or np.isnan(nrmvc):
            ffflag = 1
            break
        for ctime in ctrange:
            # bump the variables
            v_p = v_c
            mbc_p = mbc_c
            v_c, p_c = v_n, p_n
            bcs_c, mbc_c = bcs_n, mbc_n
            dfv_c = dfv_n

            # the static parts
            nfc_p = nfc_c
            nfc_c = f_vdp(appndbcs(v_c, bcs_c))

            # predict the boundary conditions
            bcs_n = getbcs(ctime, appndbcs(v_c, bcs_c), p_c, mode='abtwo')
            bfv_n, bfp_n, mbc_n = applybcs(bcs_n)

            # new values of the explicit time parts
            fv_n, fp_n = f_tdp(ctime), g_tdp(ctime)

            dfv_n, drm = dynamic_rhs(ctime, vc=v_c, memory=drm,
                                     mode='abtwo')

            rhs_n = 1/3*M@(4*v_c - v_p) \
                - (mbc_n-4/3*mbc_c+1/3*mbc_p) \
                + 2/3*dt*bfv_n \
                + 2/3*dt*(2*nfc_c-nfc_p) \
                + 2/3*dt*(fv_n + dfv_n)

            vp_n = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())

            v_n = vp_n[:NV].reshape((NV, 1))
            p_n = 1./dt*scalep*vp_n[NV:].reshape((NP, 1))

            savevp(appndbcs(v_n, bcs_n), p_n, time=ctime)

    return v_n, p_n, ffflag


def _checkuniformgrid(trange):
    dtvec = np.array(trange)[1:] - np.array(trange)[:-1]
    dotdtvec = dtvec[1:] - dtvec[:-1]
    uniformgrid = np.allclose(np.linalg.norm(dotdtvec), 0)
    if not uniformgrid:
        raise NotImplementedError()


def _onestepheun(vc=None, pc=None, tc=None, tn=None,
                 M=None, A=None, J=None,
                 scalep=1., scheme='IMEX-Euler',
                 dfv_c=None, dynamic_rhs=None, drm={},
                 # implicit_dynamic_rhs=None, mdrm={},
                 bcs_c=None, applybcs=None, appndbcs=None, getbcs=None,
                 f_tdp=None, f_vdp=None, g_tdp=None):

    NP, NV = J.shape

    dt = tn - tc
    bfv_c, _, mbc_c = applybcs(bcs_c)
    fv_c = f_tdp(tc)
    nfc_c = f_vdp(appndbcs(vc, bcs_c))
    tdfv_n, drm = dynamic_rhs(tn, vc=vc, memory=drm, mode='heunpred')
    # this was needed if there is a function only of `t`
    logging.debug('we use `tn` rather than `tc` here... ')

    # mplct_dfv_c, mdrm = implicit_dynamic_rhs(tc, vc=vc, memory=mdrm,
    #                                          mode='init')
    # mplct_tdfv, mdrm = implicit_dynamic_rhs(tc, vc=vc, memory=mdrm,
    #                                         mode='heunpred')

    tbcs = getbcs(tn, appndbcs(vc, bcs_c), pc, mode='heunpred')
    tbfv_n, tbfp_n, tmbc_n = applybcs(tbcs)
    fv_n, fp_n = f_tdp(tn), g_tdp(tn)

    # Predictor Step -- CN + explicit Euler for convection
    # norm = np.linalg.norm
    # logging.info(f'predictor: |rhs|: {norm(fv_n + tbfv_n + tdfv_n)}')

    if scheme == 'IMEX-Euler':
        tfv = M@vc \
            + dt*(fv_n + tbfv_n + tdfv_n) \
            + dt*nfc_c - (tmbc_n-mbc_c)
        tvp_n = lau.solve_sadpnt_smw(amat=M+dt*A, jmat=J, jmatT=J.T,
                                     rhsv=tfv, rhsp=fp_n+tbfp_n)
    elif scheme == 'IMEX-trpz':
        tfv = M*vc - .5*dt*A*vc \
            + .5*dt*(fv_c+fv_n + tbfv_n+bfv_c + tdfv_n+dfv_c) \
            + dt*nfc_c - (tmbc_n-mbc_c)
        tvp_n = lau.solve_sadpnt_smw(amat=M+.5*dt*A, jmat=J, jmatT=J.T,
                                     rhsv=tfv, rhsp=fp_n+tbfp_n)

    vcpc = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=J.T,
                                rhsv=fv_n, rhsp=fp_n+tbfp_n)
    nvc = vcpc[:NV, :]
    npc = vcpc[NV:, :]
    logging.info(f'stokes diff: {np.linalg.norm(vc-nvc)}')
    stkres = A@vc - fv_n
    extstkres = M@vc + dt*A@vc - tfv
    prjstkres = lau.app_prj_via_sadpnt(rhsv=stkres, amat=A, jmat=J,
                                       transposedprj=True)
    logging.info(f'prjctd stokes res: {np.linalg.norm(prjstkres)}')
    extprjstkres = lau.app_prj_via_sadpnt(rhsv=extstkres, amat=A, jmat=J,
                                          transposedprj=True)
    logging.info(f'extnd prjctd stokes res: {np.linalg.norm(extprjstkres)}')
    imexres = M@nvc + dt*A@nvc + dt*J.T@npc - dt*fv_n - M@nvc
    logging.info(f'imex res: {np.linalg.norm(imexres)}')
    tv_n = tvp_n[:NV, :]
    tp_n = 1./dt*scalep*tvp_n[NV:, :]
    rimexres = M@tv_n + dt*A@tv_n + J.T@tvp_n[NV:, :] - dt*fv_n - M@nvc
    logging.info(f'act imex res: {np.linalg.norm(rimexres)}')
    rhsdiff = np.linalg.norm(tfv - dt*fv_n-M@nvc)
    logging.info(f'rhs diff: {rhsdiff}')
    import scipy.sparse as sps
    bigamat = sps.vstack([sps.hstack([M+dt*A, J.T]),
                          sps.hstack([J, sps.csr_array((NP, NP))])])
    mvcpc = sps.linalg.spsolve(bigamat, np.vstack([tfv,
                                                   np.zeros((NP, 1))]).flatten())
    mvc = mvcpc[:NV].reshape((NV, 1))
    mpc = mvcpc[NV:].reshape((NP, 1))
    logging.info(f'diff nvc-mvc: {np.linalg.norm(nvc-mvc)}')
    logging.info(f'diff tnv-mvc: {np.linalg.norm(tv_n-mvc)}')
    mpcres = M@mvc + dt*A@mvc + J.T@mpc - tfv
    logging.info(f'mpcres: {np.linalg.norm(mpcres)}')
    # import ipdb
    # ipdb.set_trace()
    # savevp(appndbcs(tv_new, tbcs), tp_new, time=(trange[1], 'heunpred'))
    print(np.linalg.norm(tv_n - vc))

    # Corrector Step
    dfv_n, drm = dynamic_rhs(tn, vc=tv_n, memory=drm, mode='heuncorr')
    # mplct_dfv_n, mddrm = implicit_dynamic_rhs(tn, vc=tv_new,
    #                                           memory=mdrm,
    #                                           mode='heuncorr')
    tnfc_n = f_vdp(appndbcs(tv_n, tbcs))
    bcs_n = getbcs(tn, appndbcs(tv_n, tbcs), tp_n, mode='heuncorr')
    bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
    rhs_n = M*vc - (mbc_n-mbc_c) - .5*dt*A*(vc+tv_n) +\
        .5*dt*(fv_c+fv_n + bfv_n+bfv_c + dfv_n+dfv_c
               + nfc_c+tnfc_n)  # + mplct_dfv_c+mplct_dfv_n)

    # vp_new = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())
    # logging.info(f'corrector: |rhs|:
    # {norm(fv_c+fv_n + bfv_n+bfv_c + dfv_n+dfv_c + nfc_c+tnfc_n)}')
    vp_n = lau.solve_sadpnt_smw(amat=M, jmat=J, jmatT=J.T,
                                rhsv=rhs_n, rhsp=fp_n+bfp_n)
    v_n = vp_n[:NV].reshape((NV, 1))
    p_n = 1./dt*scalep*vp_n[NV:].reshape((NP, 1))

    # the implicit rhs at k=1
    # mplct_dfv_o = mplct_dfv_c
    # mplct_dfv_c, mddrm = implicit_dynamic_rhs(tn, vc=v_new, memory=mddrm)

    nfc_n = f_vdp(appndbcs(v_n, bcs_n))

    return v_n, p_n, bcs_n, bfv_n, mbc_c, mbc_n, fv_n, nfc_c, nfc_n, dfv_n, drm


def _inittimegrid(trange, ntimeslices=10):
    _checkuniformgrid(trange)
    dt = trange[1] - trange[0]
    lltr = np.array(trange[2:])
    lnts = lltr.size
    lenofts = np.floor(lnts/ntimeslices).astype(np.int)
    listofts = [lltr[k*lenofts: (k+1)*lenofts].tolist()
                for k in range(ntimeslices)]
    listofts.append(lltr[ntimeslices*lenofts:].tolist())
    return dt, listofts


def nse_include_lnrcntrllr(M=None, A=None, J=None, B=None, C=None, iniv=None,
                           hM=None, hA=None, hB=None, hC=None, hiniv=None,
                           f_vdp=None, f_tdp=None, hf_tdp=None,
                           applybcs=None, appndbcs=None, getbcs=None,
                           savevp=None):
    '''
    helper function to include a linear observer/controller

    into the linear part of the incompressible Navier-Stokes equations

    Notes
    -----

    While the matrices of the NSE follow the convention

    .. math:: M \\dot{v} + Av + J^Tp = Bu

    The controller uses standard LTI notation

    .. math:: \\dot x = \\hat{A} x + \\hat{B} u

    '''

    NP, NV = J.shape
    hNV = hA.shape[0]
    Jext = sps.hstack([J, sps.csr_matrix((NP, hNV))])
    hM = sps.eye(hNV) if hM is None else hM

    BhC = sps.csr_matrix(B@hC)
    BhC.eliminate_zeros()

    hBC = sps.csr_matrix(hB@C)
    hBC.eliminate_zeros()

    Aext = sps.vstack([sps.hstack([A, -BhC]),
                       sps.hstack([-hBC, -hA])])
    # Note the minus -- cp. the `Notes` above

    zNVhNV = sps.csr_matrix((NV, hNV))
    Mext = sps.vstack([sps.hstack([M, zNVhNV]),
                       sps.hstack([zNVhNV.T, hM])])

    inivext = np.vstack([iniv, hiniv])

    zhvec = 0*hiniv

    def _fvdpext(vvec):
        # XXX: will be called with `appendbcs`
        return np.vstack([f_vdp(vvec), zhvec])

    fvdpext = f_vdp if f_vdp is None else _fvdpext

    getbcsext = getbcs  # XXX: will be called with `appendbcs`
    savevpext = savevp  # XXX: will be called with `appendbcs`

    def ftdpext(t):
        # print('time = {0}'.format(t))
        return np.vstack([f_tdp(t), hf_tdp(t)])

    applybcsext = applybcs
    # XXX: this will fail for control Dirichlets
    # TODO: get back to this when implementing feedback Diri control

    def appndbcsext(vhvvec, ccntrlldbcvals):
        # print('y = {0}'.format(np.linalg.norm(C@vhvvec[:NV, :])))
        # print('u = {0}'.format(hC@vhvvec[NV:, :]))
        return appndbcs(vhvvec[:NV, :], ccntrlldbcvals)

    return dict(A=Aext, M=Mext, J=Jext, f_vdp=fvdpext, f_tdp=ftdpext,
                getbcs=getbcsext, applybcs=applybcsext,
                appndbcs=appndbcsext, inivel=inivext,
                savevp=savevpext)


def semi_implicit_euler(iniv=None, jmat=None, mmat=None, amat=None, rhsv=None,
                        trange=None, data_trange=None, fp=None):
    ''' integrate a NSE like system with the semi-implicit Euler method

    Mv' + Av + JTp = rhs(t, v)
          Jv       = fp


    Parameters
    ----------
    rhsv : f(t, vvec) function
        right hand side of the momentum equation
    fp : numpy array
        right hand side of the conti equation, optional, default as `fp=0`
    trange : iterable
        list or array of time instances for the integration
    data_trange : iterable
        list or array of time instances where to return the values

    Returns
    -------
    ievlist : list
        of velocity values at datatrange

    Note
    ----
    `trange` needs to be equispaced and `data_trange` needs to be a subset
    of `trange` and, in particular, `data_trange[0] == trange[0]`
    '''

    dtpt_trng = trange if data_trange is None else data_trange
    ie_dtpt_trng = (np.copy(dtpt_trng)).tolist()
    ie_dtpt_trng.pop(0)
    (NP, NV) = jmat.shape
    fpz = np.zeros((NP, 1)) if fp is None else fp
    Nts = len(trange)

    dt = trange[1] - trange[0]
    # precompute a factorization of the sad point matrix
    _, imesdpt_fctrzd = lau.solve_sadpnt_smw(amat=mmat+dt*amat, jmat=jmat,
                                             rhsv=0*iniv, return_alu=True)

    def d_impeul_increment(ct, vvec):
        # logging.info(f'IE-int ... |vc|={np.linalg.norm(vvec)}')
        iedfrhs = rhsv(ct, vvec)
        # logging.info(f'IE-int ... |rhs|={np.linalg.norm(iedfrhs)}')
        dcrhs = (mmat@vvec).reshape((-1, 1)) + dt*iedfrhs
        dslvdrhs = imesdpt_fctrzd(np.vstack([dcrhs, fpz]))
        # logging.info(f'IE-int ... |v+|={np.linalg.norm(dslvdrhs[:NV])}')
        return dslvdrhs[:NV]

    ievlist = [iniv]
    cvn = iniv
    logging.info(f'Impl. Euler integration with {Nts} time steps')
    for ct in track(trange[1:], description='semi-IE ongoing'):
        # logging.info(f'IE-int ... time={ct}:')
        cvp = cvn
        # ievlist.append(cv + impeul_increment(cv))
        cvn = d_impeul_increment(ct, cvp)
        try:
            if ct == ie_dtpt_trng[0]:
                ievlist.append(cvn)
                ie_dtpt_trng.pop(0)
            else:
                pass  # only record at data points
        except IndexError:
            logging.debug(f'ct={ct}')
            # probably the final ts not part of data trange
            pass 
    return ievlist
