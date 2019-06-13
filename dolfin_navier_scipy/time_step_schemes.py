import numpy as np

import sadptprj_riclyap_adi.lin_alg_utils as lau


def cnab(trange=None, inivel=None, inip=None, bcs_ini=[],
         M=None, A=None, J=None, nonlvfunc=None,
         fv=None, fp=None, scalep=-1.,
         getbcs=None, applybcs=None, appndbcs=None,
         savevp=None):

    dtvec = np.array(trange)[1:] - np.array(trange)[:-1]
    dotdtvec = dtvec[1:] - dtvec[:-1]
    uniformgrid = np.allclose(np.linalg.norm(dotdtvec), 0)

    if not uniformgrid:
        raise NotImplementedError()

    dt = trange[1] - trange[0]
    NP, NV = J.shape

    savevp(inivel, inip, time=trange[0])

    bcs_c = bcs_ini  # getbcs(trange[0], inivel, inip)
    bfv_c, bfp_c, mbc_c = applybcs(bcs_c)
    fv_c = fv(trange[0])
    nfc_c = nonlvfunc(inivel)

    bcs_n = getbcs(trange[1], appndbcs(inivel, bcs_ini), inip)
    bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
    fv_n, fp_n = fv(trange[1]), fp(trange[1])

    # Predictor Step -- CN + explicit Euler
    tfv = M*inivel + .5*dt*A*inivel + .5*dt*(fv_c+fv_n + bfv_n+bfv_c) \
        + dt*nfc_c - (mbc_n-mbc_c)
    tvp_new, coeffmatlu = \
        lau.solve_sadpnt_smw(amat=M+.5*dt*A, jmat=J, jmatT=J.T,
                             rhsv=tfv,
                             rhsp=fp_n+bfp_n,
                             return_alu=True)
    tv_new = tvp_new[:NV, :]

    # Corrector Step
    nfc_n = nonlvfunc(appndbcs(tv_new, bcs_n))
    bcs_n = getbcs(trange[1], appndbcs(tv_new, bcs_n), inip)
    bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
    rhs_n = M*inivel - .5*dt*A*inivel + .5*dt*(fv_c+fv_n + bfv_n+bfv_c +
                                               nfc_c+nfc_n) - (mbc_n-mbc_c)

    vp_new = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())
    v_new = vp_new[:NV].reshape((NV, 1))
    p_new = scalep*vp_new[NV:].reshape((NP, 1))

    savevp(appndbcs(v_new, bcs_n), p_new, time=trange[1])

    for ctime in trange[2:]:
        v_old, p_old = v_new, p_new
        bcs_c = bcs_n
        bfv_c, mbc_c = bfv_n, mbc_n
        fv_c = fv_n

        nfc_o = nfc_c
        nfc_c = nonlvfunc(appndbcs(v_old, bcs_c))

        bcs_n = getbcs(ctime, appndbcs(v_old, bcs_c), p_old)
        bfv_n, bfp_n, mbc_n = applybcs(bcs_n)
        fv_n, fp_n = fv(ctime), fp(ctime)

        rhs_n = M*v_old - .5*dt*A*v_old + .5*dt*(fv_c+fv_n + bfv_n+bfv_c) +\
            1.5*dt*nfc_c-.5*dt*nfc_o - (mbc_n-mbc_c)

        vp_new = coeffmatlu(np.vstack([rhs_n, fp_n+bfp_n]).flatten())
        v_new = vp_new[:NV].reshape((NV, 1))
        p_new = scalep*vp_new[NV:].reshape((NP, 1))

        savevp(appndbcs(v_new, bcs_n), p_new, time=ctime)

        nfc_o = nfc_c
        bfv_c, mbc_c = bfv_n, mbc_n
        fv_c = fv_n

    return v_new, p_new
