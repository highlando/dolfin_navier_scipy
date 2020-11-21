import numpy as np

import sadptprj_riclyap_adi.lin_alg_utils as lau

# from dolfin_navier_scipy.residual_checks import get_imex_res
# from dolfin_navier_scipy.dolfin_to_sparrays import expand_vp_dolfunc


def cnab(trange=None, inivel=None, inip=None, bcs_ini=[],
         M=None, A=None, J=None,
         f_vdp=None,
         f_tdp=None, g_tdp=None,
         scalep=-1.,
         getbcs=None, applybcs=None, appndbcs=None,
         savevp=None, dynamic_rhs=None, dynamic_rhs_memory={},
         ntimeslices=10, verbose=True):

    dtvec = np.array(trange)[1:] - np.array(trange)[:-1]
    dotdtvec = dtvec[1:] - dtvec[:-1]
    uniformgrid = np.allclose(np.linalg.norm(dotdtvec), 0)
    if verbose:
        import time

    if not uniformgrid:
        raise NotImplementedError()

    dt = trange[1] - trange[0]
    NP, NV = J.shape
    zerorhs = np.zeros((NV, 1))

    if dynamic_rhs is None:
        def dynamic_rhs(t, vc=None, memory={}, mode=None):
            return zerorhs, memory

    drm = dynamic_rhs_memory

    savevp(appndbcs(inivel, bcs_ini), inip, time=trange[0])

    bcs_c = bcs_ini  # getbcs(trange[0], inivel, inip)
    bfv_c, bfp_c, mbc_c = applybcs(bcs_c)
    fv_c = f_tdp(trange[0])
    nfc_c = f_vdp(appndbcs(inivel, bcs_ini))
    dfv_c, drm = dynamic_rhs(trange[0], vc=inivel, memory=drm, mode='init')
    tdfv, drm = dynamic_rhs(trange[0], vc=inivel, memory=drm, mode='heunpred')

    tbcs = getbcs(trange[1], appndbcs(inivel, bcs_ini), inip, mode='heunpred')
    tbfv, tbfp, tmbc = applybcs(tbcs)
    fv_n, fp_n = f_tdp(trange[1]), g_tdp(trange[1])

    # Predictor Step -- CN + explicit Euler
    tfv = M*inivel - .5*dt*A*inivel \
        + .5*dt*(fv_c+fv_n + tbfv+bfv_c + tdfv+dfv_c) \
        + dt*nfc_c - (tmbc-mbc_c)

    tvp_new, coeffmatlu = \
        lau.solve_sadpnt_smw(amat=M+.5*dt*A, jmat=J, jmatT=J.T,
                             rhsv=tfv,
                             rhsp=fp_n+tbfp,
                             return_alu=True)
    tv_new = tvp_new[:NV, :]
    tp_new = 1./dt*scalep*tvp_new[NV:, :]
    # savevp(appndbcs(tv_new, tbcs), tp_new, time=(trange[1], 'heunpred'))

    # Corrector Step
    dfv_n, drm = dynamic_rhs(trange[1], vc=tv_new, memory=drm, mode='heuncorr')
    nfc_n = f_vdp(appndbcs(tv_new, tbcs))
    bcs_n = getbcs(trange[1], appndbcs(tv_new, tbcs), tp_new, mode='heuncorr')
    bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
    rhs_n = M*inivel - .5*dt*A*inivel - (mbc_n-mbc_c) +\
        .5*dt*(fv_c+fv_n + bfv_n+bfv_c + dfv_n+tdfv + nfc_c+nfc_n)

    vp_new = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())
    v_new = vp_new[:NV].reshape((NV, 1))
    p_new = 1./dt*scalep*vp_new[NV:].reshape((NP, 1))

    savevp(appndbcs(v_new, bcs_n), p_new, time=trange[1])

    lltr = np.array(trange[2:])
    lnts = lltr.size
    lenofts = np.floor(lnts/ntimeslices).astype(np.int)
    listofts = [lltr[k*lenofts: (k+1)*lenofts].tolist()
                for k in range(ntimeslices)]
    listofts.append(lltr[ntimeslices*lenofts:].tolist())

    verbose = True
    for kck, ctrange in enumerate(listofts):
        if verbose:
            print('time-stepping {0}/{1} complete -- @runtime {2:.1f}'.
                  format(kck, ntimeslices, time.process_time()))
        for ctime in ctrange:
            v_old, p_old = v_new, p_new
            bcs_c = bcs_n
            bfv_c, mbc_c = bfv_n, mbc_n
            fv_c = fv_n
            dfv_c = dfv_n

            nfc_o = nfc_c
            nfc_c = f_vdp(appndbcs(v_old, bcs_c))

            bcs_n = getbcs(ctime, appndbcs(v_old, bcs_c), p_old, mode='abtwo')
            bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
            fv_n, fp_n = f_tdp(ctime), g_tdp(ctime)
            dfv_n, drm = dynamic_rhs(ctime, vc=v_old, memory=drm,
                                     mode='abtwo')

            rhs_n = M*v_old - .5*dt*A*v_old + 1.5*dt*nfc_c-.5*dt*nfc_o \
                - (mbc_n-mbc_c) \
                + .5*dt*(fv_c+fv_n + bfv_n+bfv_c + dfv_n+dfv_c)

            vp_new = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())

            v_new = vp_new[:NV].reshape((NV, 1))
            p_new = 1./dt*scalep*vp_new[NV:].reshape((NP, 1))

            savevp(appndbcs(v_new, bcs_n), p_new, time=ctime)

    return v_new, p_new


def get_heunab_lti(hb=None, ha=None, hc=None, inihx=None, drift=None):
    """ realizes the Heun/AB2 discretization of a linear observer

    ```
    hx' = hA*hx + hb*y
     u  = hc*hx
    ```

    e.g. with `hb=C*C` and `hc=BB*X` to get output based feedback actuation
    """

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
