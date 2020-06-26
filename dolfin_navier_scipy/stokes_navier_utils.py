import numpy as np
import scipy.sparse as sps
import os
import glob
import time
# import sys
# import copy
import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou


__all__ = ['get_datastr_snu',
           'get_v_conv_conts',
           'solve_nse',
           'solve_steadystate_nse',
           'get_pfromv']


def get_datastr_snu(time=None, meshp=None, nu=None, Nts=None, data_prfx='',
                    semiexpl=False):
    sestr = '' if not semiexpl else '_semexp'
    nustr = '_nuNone' if nu is None else '_nu{0:.3e}'.format(nu)
    ntsstr = '_NtsNone' if Nts is None else '_Nts{0}'.format(Nts)
    timstr = 'timeNone' if time is None or isinstance(time, str) else \
        'time{0:.5e}'.format(time)
    mshstr = '_mesh{0}'.format(meshp)

    return data_prfx + timstr + nustr + mshstr + ntsstr + sestr

    # if time is None or isinstance(time, str):
    #     return (data_prfx + 'time{0}_nu{1:.3e}_mesh{2}_Nts{3}'.
    #             format(time, nu, meshp, Nts) + sestr)
    # else:
    #     return (data_prfx + 'time{0:.5e}_nu{1:.3e}_mesh{2}_Nts{3}'.
    #             format(time, nu, meshp, Nts) + sestr)


def get_v_conv_conts(vvec=None, V=None,
                     invinds=None, dbcvals=[], dbcinds=[],
                     semi_explicit=False, Picard=False, retparts=False):
    """ get and condense the linearized convection

    to be used in a Newton scheme

    .. math::

        (u \\cdot \\nabla) u \\to (u_0 \\cdot \\nabla) u + \
            (u \\cdot \\nabla) u_0 - (u_0 \\cdot \\nabla) u_0

    or in a Picard scheme

    .. math::

        (u \\cdot \\nabla) u \\to (u_0 \\cdot \\nabla) u

    Parameters
    ----------
    vvec : (N,1) ndarray
        convection velocity
    V : dolfin.VectorFunctionSpace
        FEM space of the velocity
    invinds : (N,) ndarray or list
        indices of the inner nodes
    Picard : Boolean
        whether Picard linearization is applied, defaults to `False`
    semi_explicit: Boolean, optional
        whether to return minus the convection vector, and zero convmats
        as needed for semi-explicit integration, defaults to `False`
    retparts : Boolean, optional
        whether to return both components of the matrices
        and contributions to the rhs through the boundary conditions,
        defaults to `False`

    Returns
    -------
    convc_mat : (N,N) sparse matrix
        representing the linearized convection at the inner nodes
    rhs_con : (N,1) array
        representing :math:`(u_0 \\cdot \\nabla )u_0` at the inner nodes
    rhsv_conbc : (N,1) ndarray
        representing the boundary conditions

    Note
    ----
    If `vvec` has the boundary conditions already included, than the provided
    `dbcinds`, `dbcvals` are only used to condense the matrices

    """

    vfun = dolfin.Function(V)
    if len(vvec) == V.dim():
        ve = vvec
    else:
        ve = np.full((V.dim(), ), np.nan)
        ve[invinds] = vvec.flatten()
        for k, cdbcinds in enumerate(dbcinds):
            ve[cdbcinds] = dbcvals[k]

    vfun.vector().set_local(ve)

    if semi_explicit:
        rhs_con = dts.get_convvec(V=V, u0_dolfun=vfun, invinds=invinds,
                                  dbcinds=dbcinds, dbcvals=dbcvals)

        return 0., -rhs_con, 0.

    N1, N2, rhs_con = dts.get_convmats(u0_dolfun=vfun, V=V, invinds=invinds,
                                       dbcinds=dbcinds, dbcvals=dbcvals)

    _cndnsmts = dts.condense_velmatsbybcs

    if Picard:
        convc_mat, rhsv_conbc = _cndnsmts(N1, invinds=invinds,
                                          dbcinds=dbcinds, dbcvals=dbcvals)
        # return convc_mat, rhs_con[invinds, ], rhsv_conbc
        return convc_mat, None, rhsv_conbc

    elif retparts:
        picrd_convc_mat, picrd_rhsv_conbc = \
            _cndnsmts(N1, invinds=invinds, dbcinds=dbcinds, dbcvals=dbcvals)
        anti_picrd_convc_mat, anti_picrd_rhsv_conbc = \
            _cndnsmts(N2, invinds=invinds, dbcinds=dbcinds, dbcvals=dbcvals)
        return ((picrd_convc_mat, anti_picrd_convc_mat),
                rhs_con[invinds, ],
                (picrd_rhsv_conbc, anti_picrd_rhsv_conbc))

    else:
        convc_mat, rhsv_conbc = _cndnsmts(N1+N2, invinds=invinds,
                                          dbcinds=dbcinds, dbcvals=dbcvals)

        return convc_mat, rhs_con[invinds, ], rhsv_conbc


def m_innerproduct(M, v1, v2=None):
    """ inner product with a spd sparse matrix

    """
    if v2 is None:
        v2 = v1  # in most cases, we want to compute the norm

    return np.dot(v1.T, M*v2)


def _localizecdbinds(cdbinds, V, invinds):
    """ find the local indices of the control dirichlet boundaries

    the given control dirichlet boundaries were indexed w.r.t. the
    full space `V`. Here, in the matrices, we have already
    resolved the constant Dirichlet bcs
    """
    allinds = np.arange(V.dim())
    redcdallinds = allinds[invinds]
    # now: find the positions of the control dbcs in the reduced
    # index vector
    lclinds = np.searchsorted(redcdallinds, cdbinds, side='left')
    return lclinds


def _comp_cntrl_bcvals(diricontbcvals=[], diricontfuncs=[], mode=None,
                       diricontfuncmems=[], time=None, vel=None, p=None, **kw):
    cntrlldbcvals = []
    try:
        for k, cdbbcv in enumerate(diricontbcvals):
            ccntrlfunc = diricontfuncs[k]
            try:
                cntrlval, diricontfuncmems[k] = \
                    ccntrlfunc(time, vel=vel, p=p, mode=mode,
                               memory=diricontfuncmems[k])
            except TypeError:
                cntrlval, diricontfuncmems[k] = \
                    ccntrlfunc(time, vel=vel, p=p,
                               memory=diricontfuncmems[k])

            ccntrlldbcvals = [cntrlval*bcvl for bcvl in cdbbcv]
            cntrlldbcvals.extend(ccntrlldbcvals)
    except TypeError:
        pass  # no controls applied
    return cntrlldbcvals


def _cntrl_stffnss_rhs(loccntbcinds=None, cntrlldbcvals=None,
                       vvec=None, A=None, J=None, **kw):

    if vvec is not None:
        ccfv = dts.condense_velmatsbybcs(A, invinds=loccntbcinds,
                                         vwithbcs=vvec, get_rhs_only=True)
        ccfp = dts.condense_velmatsbybcs(J, invinds=loccntbcinds,
                                         vwithbcs=vvec, get_rhs_only=True,
                                         columnsonly=True)
        return ccfv, ccfp

    crhsdct = dts.condense_sysmatsbybcs(dict(A=A, J=J),
                                        dbcvals=cntrlldbcvals,
                                        dbcinds=loccntbcinds,
                                        get_rhs_only=True)
    return crhsdct['fv'], crhsdct['fp']


def _attach_cntbcvals(vvec, globbcinds=None, dbcvals=None,
                      globbcinvinds=None, invinds=None, NV=None):
    auxv = np.full((NV, ), np.nan)
    auxv[globbcinvinds] = vvec
    auxv[globbcinds] = dbcvals
    return auxv[invinds]


def solve_steadystate_nse(A=None, J=None, JT=None, M=None,
                          fv=None, fp=None,
                          V=None, Q=None, invinds=None, diribcs=None,
                          dbcvals=None, dbcinds=None,
                          diricontbcinds=None, diricontbcvals=None,
                          diricontfuncs=None, diricontfuncmems=None,
                          return_vp=False, ppin=None,
                          return_nwtnupd_norms=False,
                          N=None, nu=None,
                          vel_pcrd_stps=10, vel_pcrd_tol=1e-4,
                          vel_nwtn_stps=20, vel_nwtn_tol=5e-15,
                          clearprvdata=False,
                          useolddata=False,
                          vel_start_nwtn=None,
                          get_datastring=None,
                          data_prfx='',
                          paraviewoutput=False,
                          save_data=True,
                          save_intermediate_steps=False,
                          vfileprfx='', pfileprfx='',
                          verbose=True,
                          **kw):

    """
    Solution of the steady state nonlinear NSE Problem

    using Newton's scheme. If no starting value is provided, the iteration
    is started with the steady state Stokes solution.

    Parameters
    ----------
    A : (N,N) sparse matrix
        stiffness matrix aka discrete Laplacian, note the sign!
    M : (N,N) sparse matrix
        mass matrix
    J : (M,N) sparse matrix
        discrete divergence operator
    JT : (N,M) sparse matrix, optional
        discrete gradient operator, set to J.T if not provided
    fv, fp : (N,1), (M,1) ndarrays
        right hand sides restricted via removing the boundary nodes in the
        momentum and the pressure freedom in the continuity equation
    ppin : {int, None}, optional
        which dof of `p` is used to pin the pressure, defaults to `None`
    dbcinds: list, optional
        indices of the Dirichlet boundary conditions
    dbcvals: list, optional
        values of the Dirichlet boundary conditions (as listed in `dbcinds`)
    diricontbcinds: list, optional
        list of dirichlet indices that are to be controlled
    diricontbcvals: list, optional
        list of the vals corresponding to `diricontbcinds`
    diricontfuncs: list, optional
        list like `[ufunc]` where `ufunc: (t, v) -> u` where `u` is used to
        scale the corresponding `diricontbcvals`
    return_vp : boolean, optional
        whether to return also the pressure, defaults to `False`
    vel_pcrd_stps : int, optional
        Number of Picard iterations when computing a starting value for the
        Newton scheme, cf. Elman, Silvester, Wathen: *FEM and fast iterative
        solvers*, 2005, defaults to `100`
    vel_pcrd_tol : real, optional
        tolerance for the size of the Picard update, defaults to `1e-4`
    vel_nwtn_stps : int, optional
        Number of Newton iterations, defaults to `20`
    vel_nwtn_tol : real, optional
        tolerance for the size of the Newton update, defaults to `5e-15`

    Returns:
    ---
    vel_k : (N, 1) ndarray
        the velocity vector, if not `return_vp`, else
    (v, p) : tuple
        of the velocity and the pressure vector
    norm_nwtnupd_list : list, on demand
        list of the newton upd errors
    """

    import sadptprj_riclyap_adi.lin_alg_utils as lau

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if JT is None:
        JT = J.T

    NV = J.shape[1]
    dbcinds, dbcvals = dts.unroll_dlfn_dbcs(diribcs, bcinds=dbcinds,
                                            bcvals=dbcvals)

    norm_nwtnupd_list = []
    # a dict to be passed to the get_datastring function
    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=None, data_prfx=data_prfx)

    if clearprvdata:
        cdatstr = get_datastring(**datastrdict)
        for fname in glob.glob(cdatstr + '*__vel*'):
            os.remove(fname)

    if useolddata:
        try:
            cdatstr = get_datastring(**datastrdict)

            norm_nwtnupd = dou.load_npa(cdatstr + '__norm_nwtnupd')
            vel_k = dou.load_npa(cdatstr + '__vel')
            norm_nwtnupd_list.append(norm_nwtnupd)

            if verbose:
                print('found vel files')
                print('norm of last Nwtn update: {0}'.format(norm_nwtnupd))
                print('... loaded from ' + cdatstr)
            if np.atleast_1d(norm_nwtnupd)[0] is None:
                norm_nwtnupd = None
                pass  # nothing useful found

            elif norm_nwtnupd < vel_nwtn_tol:
                if not return_vp:
                    return vel_k, norm_nwtnupd_list
                else:
                    pfv = get_pfromv(v=vel_k[:NV, :], V=V,
                                     M=M, A=A, J=J, fv=fv,
                                     dbcinds=dbcinds, dbcvals=dbcvals,
                                     invinds=invinds)
                    return (np.vstack([vel_k, pfv]), norm_nwtnupd_list)

        except IOError:
            if verbose:
                print('no old velocity data found')
            norm_nwtnupd = None

    else:
        # we start from scratch
        norm_nwtnupd = None

    if paraviewoutput:
        cdatstr = get_datastring(**datastrdict)
        vfile = dolfin.File(vfileprfx+'__steadystates.pvd')
        pfile = dolfin.File(pfileprfx+'__steadystates.pvd')
        prvoutdict = dict(V=V, Q=Q, vfile=vfile, pfile=pfile,
                          invinds=invinds, diribcs=diribcs, ppin=ppin,
                          dbcinds=dbcinds, dbcvals=dbcvals,
                          vp=None, t=None, writeoutput=True)
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' later

    NV = A.shape[0]

    loccntbcinds, glbcntbcinds = [], []
    if diricontbcinds is None or diricontbcinds == []:
        cmmat, camat, cj, cjt, cfv, cfp = M, A, J, JT, fv, fp
        cnv = NV
        dbcntinvinds = invinds
    else:
        for cdbidbv in diricontbcinds:
            localbcinds = (_localizecdbinds(cdbidbv, V, invinds)).tolist()
            loccntbcinds.extend(localbcinds)  # adding the boundary inds
            glbcntbcinds.extend(cdbidbv)

        dbcntinvinds = np.setdiff1d(invinds, glbcntbcinds).astype(np.int32)
        locdbcntinvinds = (_localizecdbinds(dbcntinvinds, V, invinds)).tolist()
        cmmat = M[locdbcntinvinds, :][:, locdbcntinvinds]
        camat = A[locdbcntinvinds, :][:, locdbcntinvinds]
        cjt = JT[locdbcntinvinds, :]
        cj = J[:, locdbcntinvinds]
        cnv = cmmat.shape[0]
        cfp = fp
        cfv = fv[locdbcntinvinds]

    cntrlmatrhsdict = {'A': A, 'J': J,  # 'fv': fv, 'fp': fp,
                       'loccntbcinds': loccntbcinds,
                       'diricontbcvals': diricontbcvals,
                       'diricontfuncs': diricontfuncs,
                       'diricontfuncmems': diricontfuncmems
                       }

    def _appbcs(vvec, ccntrlldbcvals):
        return dts.append_bcs_vec(vvec, vdim=V.dim(), invinds=dbcntinvinds,
                                  bcinds=[dbcinds, glbcntbcinds],
                                  bcvals=[dbcvals, ccntrlldbcvals])

    if vel_start_nwtn is None:
        cdbcvals_c = _comp_cntrl_bcvals(time=None, vel=None, p=None,
                                        mode='init',
                                        **cntrlmatrhsdict)
        ccfv, ccfp = _cntrl_stffnss_rhs(cntrlldbcvals=cdbcvals_c,
                                        **cntrlmatrhsdict)

        vp_stokes = lau.solve_sadpnt_smw(amat=camat, jmat=cj, jmatT=cjt,
                                         rhsv=cfv+ccfv, rhsp=cfp+ccfp)
        vp_stokes[cnv:] = -vp_stokes[cnv:]
        # pressure was flipped for symmetry

        # save the data
        cdatstr = get_datastring(**datastrdict)

        if save_data:
            dou.save_npa(vp_stokes[:cnv, ], fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_stokes,
                               dbcinds=[dbcinds, glbcntbcinds],
                               dbcvals=[dbcvals, cdbcvals_c],
                               invinds=dbcntinvinds))
        dou.output_paraview(**prvoutdict)

        # Stokes solution as starting value
        vp_k = vp_stokes
        vel_k = vp_stokes[:cnv, ]
        p_k = vp_stokes[cnv:, ]

    else:
        cdbcvals_c = vel_start_nwtn[glbcntbcinds, :]
        vel_k = vel_start_nwtn[dbcntinvinds, :]
        # print('TODO: what about the ini pressure')
        p_k = np.zeros((J.shape[0], 1))
        vpsnwtn = np.vstack([vel_k, p_k])
        prvoutdict.update(dict(vp=vpsnwtn,
                               dbcinds=[dbcinds, glbcntbcinds],
                               dbcvals=[dbcvals, cdbcvals_c],
                               invinds=dbcntinvinds))
        dou.output_paraview(**prvoutdict)

    # Picard iterations for a good starting value for Newton
    for k in range(vel_pcrd_stps):

        cdbcvals_n = _comp_cntrl_bcvals(vel=_appbcs(vel_k, cdbcvals_c),
                                        p=p_k, **cntrlmatrhsdict)

        ccfv_n, ccfp_n = _cntrl_stffnss_rhs(cntrlldbcvals=cdbcvals_n,
                                            **cntrlmatrhsdict)

        # use the old v-bcs to compute the convection
        # TODO: actually we only need Picard -- do some fine graining in dts
        N1, N2, rhscnv = dts.get_convmats(u0_vec=_appbcs(vel_k, cdbcvals_c),
                                          V=V)

        # apply the new v-bcs
        pcrdcnvmat, rhsv_conbc = dts.\
            condense_velmatsbybcs(N1, invinds=dbcntinvinds,
                                  dbcinds=[dbcinds, glbcntbcinds],
                                  dbcvals=[dbcvals, cdbcvals_n])

        vp_k = lau.solve_sadpnt_smw(amat=camat+pcrdcnvmat, jmat=cj, jmatT=cjt,
                                    rhsv=cfv+ccfv_n+rhsv_conbc,
                                    rhsp=cfp+ccfp_n)

        normpicupd = np.sqrt(m_innerproduct(cmmat, vel_k-vp_k[:cnv, ]))[0]

        if verbose:
            print('Picard iteration: {0} -- norm of update: {1}'.
                  format(k+1, normpicupd))

        vel_k = vp_k[:cnv, ]
        vp_k[cnv:] = -vp_k[cnv:]
        # pressure was flipped for symmetry

        if normpicupd < vel_pcrd_tol:
            break

    # Newton iteration

    for vel_newtk, k in enumerate(range(vel_nwtn_stps)):

        cdatstr = get_datastring(**datastrdict)

        cdbcvals_n = _comp_cntrl_bcvals(vel=_appbcs(vel_k, cdbcvals_c),
                                        p=p_k, **cntrlmatrhsdict)

        ccfv_n, ccfp_n = _cntrl_stffnss_rhs(cntrlldbcvals=cdbcvals_n,
                                            **cntrlmatrhsdict)
        (convc_mat, rhs_con, rhsv_conbc) = \
            get_v_conv_conts(vvec=_appbcs(vel_k, cdbcvals_c), V=V,
                             invinds=dbcntinvinds,
                             dbcinds=[dbcinds, glbcntbcinds],
                             dbcvals=[dbcvals, cdbcvals_n])

        vp_k = lau.solve_sadpnt_smw(amat=camat+convc_mat, jmat=cj, jmatT=cjt,
                                    rhsv=cfv+ccfv_n+rhs_con+rhsv_conbc,
                                    rhsp=cfp+ccfp_n)

        norm_nwtnupd = np.sqrt(m_innerproduct(cmmat, vel_k - vp_k[:cnv, :]))[0]
        vel_k = vp_k[:cnv, ]
        vp_k[cnv:] = -vp_k[cnv:]
        p_k = vp_k[cnv:, ]
        cdbcvals_c = cdbcvals_n
        # pressure was flipped for symmetry
        if verbose:
            print('Steady State NSE: Newton iteration: {0}'.format(vel_newtk) +
                  '-- norm of update: {0}'.format(norm_nwtnupd))

        if save_data:
            dou.save_npa(vel_k, fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k))  # , dbcvals=[dbcvals, cdbcvals_n]))
        # TODO: werden die wirklich implicit ubgedated?
        dou.output_paraview(**prvoutdict)

        if norm_nwtnupd < vel_nwtn_tol:
            break

    else:
        if vel_nwtn_stps == 0:
            print('No Newton steps = steady state probably not well converged')
        else:
            raise UserWarning('Steady State NSE: Newton has not converged')

    if save_data:
        dou.save_npa(norm_nwtnupd, cdatstr + '__norm_nwtnupd')

    prvoutdict.update(dict(vp=vp_k))  # , dbcvals=[dbcvals, cntrlldbcvals]))
    dou.output_paraview(**prvoutdict)

    # savetomatlab = True
    # if savetomatlab:
    #     export_mats_to_matlab(E=None, A=None, matfname='matexport')

    vwc = _appbcs(vel_k, cdbcvals_c).reshape((V.dim(), 1))
    if return_vp:
        retthing = (vwc, vp_k[cnv:, :])
    else:
        retthing = vwc
    if return_nwtnupd_norms:
        return retthing, norm_nwtnupd_list
    else:
        return retthing


def solve_nse(A=None, M=None, J=None, JT=None,
              fv=None, fp=None,
              fvtd=None, fvss=0.,
              # TODO: fv_tmdp=None, fv_tmdp_params={}, fv_tmdp_memory=None,
              iniv=None, inip=None, lin_vel_point=None,
              stokes_flow=False,
              trange=None,
              t0=None, tE=None, Nts=None,
              V=None, Q=None, invinds=None, diribcs=None,
              dbcinds=None, dbcvals=None,
              diricontbcinds=None, diricontbcvals=None,
              diricontfuncs=None, diricontfuncmems=None,
              N=None, nu=None,
              ppin=None,
              closed_loop=False,
              static_feedback=False, stat_fb_dict={},
              dynamic_feedback=False, dyn_fb_dict={},
              feedbackthroughdict=None,
              return_vp=False,
              b_mat=None, cv_mat=None,
              vel_nwtn_stps=20, vel_nwtn_tol=5e-15,
              nsects=1, loc_nwtn_tol=5e-15, loc_pcrd_stps=True,
              addfullsweep=False,
              vel_pcrd_stps=4,
              krylov=None, krpslvprms={}, krplsprms={},
              clearprvdata=False,
              useolddata=False,
              get_datastring=None,
              data_prfx='',
              paraviewoutput=False,
              plttrange=None,
              vfileprfx='', pfileprfx='',
              return_dictofvelstrs=False,
              return_dictofpstrs=False,
              dictkeysstr=False,
              treat_nonl_explct=False, no_data_caching=True,
              return_final_vp=False,
              return_as_list=False, return_vp_dict=False,
              verbose=True,
              start_ssstokes=False,
              **kw):
    """
    solution of the time-dependent nonlinear Navier-Stokes equation

    .. math::
        M\\dot v + Av + N(v)v + J^Tp = f_v \n
        Jv = f_p

    using a Newton scheme in function space, i.e. given :math:`v_k`,
    we solve for the update like

    .. math::
        M\\dot v + Av + N(v_k)v + N(v)v_k + J^Tp = N(v_k)v_k + f,

    and trapezoidal rule in time. To solve an *Oseen* system (linearization
    about a steady state) or a *Stokes* system, set the number of Newton
    steps to one and provide a linearization point and an initial value.


    Parameters
    ----------
    lin_vel_point : dictionary, optional
        contains the linearization point for the first Newton iteration

         * Steady State: {{`None`: 'path_to_nparray'}, {'None': nparray}}
         * Newton: {`t`: 'path_to_nparray'}

        defaults to `None`
    dictkeysstr : boolean, optional
        whether the `keys` of the result dictionaries are strings instead \
        of floats, defaults to `False`
    fvtd : callable f(t), optional
        time dependend right hand side in momentum equation
    fvss : array, optional
        right hand side in momentum for steady state computation
    fv_tmdp : callable f(t, v, dict), optional
        time-dependent part of the right-hand side, set to zero if None
    fv_tmdp_params : dictionary, optional
        dictionary of parameters to be passed to `fv_tmdp`, defaults to `{}`
    fv_tmdp_memory : dictionary, optional
        memory of the function
    dbcinds: list, optional
        indices of the Dirichlet boundary conditions
    dbcvals: list, optional
        values of the Dirichlet boundary conditions (as listed in `dbcinds`)
    diricontbcinds: list, optional
        list of dirichlet indices that are to be controlled
    diricontbcvals: list, optional
        list of the vals corresponding to `diricontbcinds`
    diricontfuncs: list, optional
        list like `[ufunc]` where `ufunc: (t, v) -> u` where `u` is used to
        scale the corresponding `diricontbcvals`
    # output_includes_bcs : boolean, optional
    #     whether append the boundary nodes to the computed and stored \
    #     velocities, defaults to `False`
    krylov : {None, 'gmres'}, optional
        whether or not to use an iterative solver, defaults to `None`
    krpslvprms : dictionary, optional
        v specify parameters of the linear solver for use in Krypy, e.g.,

          * initial guess
          * tolerance
          * number of iterations

        defaults to `None`
    krplsprms : dictionary, optional
        parameters to define the linear system like

          * preconditioner

    ppin : {int, None}, optional
        which dof of `p` is used to pin the pressure, defaults to `None`
    stokes_flow : boolean, optional
        whether to consider the Stokes linearization, defaults to `False`
    start_ssstokes : boolean, optional
        for your convenience, compute and use the steady state stokes solution
        as initial value, defaults to `False`
    treat_nonl_explct= string, optional
        whether to treat the nonlinearity explicitly, defaults to `False`
    nsects: int, optional
        in how many segments the trange is split up. (The newton iteration
        will be confined to the segments and, probably, converge faster than
        the global iteration), defaults to `1`
    loc_nwtn_tol: float, optional
        tolerance for the newton iteration on the segments,
        defaults to `1e-15`
    loc_pcrd_stps: boolean, optional
        whether to init with `vel_pcrd_stps` Picard steps on every section,
        if `False`, Picard iterations are performed only on the first section,
        defaults to `True`
    addfullsweep: boolean, optional
        whether to compute the newton iteration on the full `trange`,
        useful to check and for the plots, defaults to `False`
    cv_mat: (Ny, Nv) sparse array, optional
        output matrix for velocity outputs, needed, e.g., for output dependent
        feedback control, defaults to `None`
    dynamic_feedback: boolean
        whether to apply dynamic feedback, defaults to `False`
    dyn_fb_dict, dictionary
        that defines the dynamic observer via the keys
          * `ha` observer dynamic matrix
          * `hb` observer input matrix
          * `hc` observer output matrix
          * `drift` observer drift term, e.g., for nonzero setpoints


    Returns
    -------
    dictofvelstrs : dictionary, on demand
        dictionary with time `t` as keys and path to velocity files as values

    dictofpstrs : dictionary, on demand
        dictionary with time `t` as keys and path to pressure files as values

    vellist : list, on demand
        list of the velocity solutions

    """
    import sadptprj_riclyap_adi.lin_alg_utils as lau

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if trange is None:
        trange = np.linspace(t0, tE, Nts+1)

    if treat_nonl_explct and lin_vel_point is not None:
        raise UserWarning('cant use `lin_vel_point` ' +
                          'and explicit treatment of the nonlinearity')

    dbcinds, dbcvals = dts.unroll_dlfn_dbcs(diribcs, bcinds=dbcinds,
                                            bcvals=dbcvals)

    loccntbcinds, glbcntbcinds = [], []
    if diricontbcinds is None or diricontbcinds == []:
        dbcntinvinds = invinds
    else:
        for k, cdbidbv in enumerate(diricontbcinds):
            localbcinds = (_localizecdbinds(cdbidbv, V, invinds)).tolist()
            loccntbcinds.extend(localbcinds)  # adding the boundary inds
            glbcntbcinds.extend(cdbidbv)
        dbcntinvinds = np.setdiff1d(invinds, glbcntbcinds).astype(np.int32)

    locinvinds = (_localizecdbinds(dbcntinvinds, V, invinds)).tolist()
    cmmat = M[locinvinds, :][:, locinvinds]
    camat = A[locinvinds, :][:, locinvinds]
    cjt = JT[locinvinds, :]
    cj = J[:, locinvinds]
    cfv = fv[locinvinds]
    cfp = fp

    cntrlmatrhsdict = {'A': A, 'J': J,
                       'loccntbcinds': loccntbcinds,
                       'diricontbcvals': diricontbcvals,
                       'diricontfuncs': diricontfuncs,
                       'diricontfuncmems': diricontfuncmems
                       }

    cnv = dbcntinvinds.size
    NP = J.shape[0]
    fv = np.zeros((cnv, 1)) if fv is None else fv
    fp = np.zeros((NP, 1)) if fp is None else fp

    prvoutdict = dict(V=V, Q=Q, vp=None, t=None,
                      dbcinds=[dbcinds, glbcntbcinds],
                      dbcvals=[dbcvals],
                      invinds=dbcntinvinds, ppin=ppin,
                      tfilter=plttrange, writeoutput=paraviewoutput)

    # ## XXX: looks like this needs treatment
    # if return_dictofpstrs:
    #     gpfvd = dict(V=V, M=M, A=A, J=J, fv=fv, fp=fp,
    #                  dbcinds=dbcinds, dbcvals=dbcvals, invinds=invinds)

    # if fv_tmdp is None:
    #     def fv_tmdp(time=None, curvel=None, **kw):
    #         return np.zeros((cnv, 1)), None

# ----- #
# chap: # the initial value
# ----- #

    if iniv is None:
        if start_ssstokes:
            inicdbcvals = _comp_cntrl_bcvals(time=trange[0], vel=None, p=None,
                                             mode='stokes', **cntrlmatrhsdict)
            ccfv, ccfp = _cntrl_stffnss_rhs(cntrlldbcvals=inicdbcvals,
                                            **cntrlmatrhsdict)
            # Stokes solution as starting value
            vp_stokes =\
                lau.solve_sadpnt_smw(amat=camat, jmat=cj, jmatT=cjt,
                                     rhsv=cfv+ccfv+fvss,
                                     krylov=krylov, krpslvprms=krpslvprms,
                                     krplsprms=krplsprms, rhsp=cfp+ccfp)
            iniv = vp_stokes[:cnv]
        else:
            raise ValueError('No initial value given')
    else:
        inicdbcvals = (iniv[glbcntbcinds].flatten()).tolist()
        iniv = iniv[dbcntinvinds]
        ccfv, ccfp = _cntrl_stffnss_rhs(cntrlldbcvals=inicdbcvals,
                                        **cntrlmatrhsdict)
        # # initialization
        # _comp_cntrl_bcvals(time=trange[0], vel=iniv, p=inip,
        #                    mode='init', **cntrlmatrhsdict)
    if inip is None:
        inip = get_pfromv(v=iniv, V=V, M=cmmat, A=cmmat, J=cj,
                          fv=cfv+ccfv+fvss, fp=cfp+ccfp,
                          dbcinds=[dbcinds, glbcntbcinds],
                          dbcvals=[dbcvals, inicdbcvals], invinds=dbcntinvinds)

    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=trange.size-1, data_prfx=data_prfx,
                       semiexpl=treat_nonl_explct)

    if return_as_list:
        clearprvdata = True  # we want the results at hand
    if clearprvdata:
        datastrdict['time'] = '*'
        cdatstr = get_datastring(**datastrdict)
        for fname in glob.glob(cdatstr + '__vel*'):
            os.remove(fname)
        for fname in glob.glob(cdatstr + '__p*'):
            os.remove(fname)

    if return_dictofvelstrs or return_dictofpstrs:
        no_data_caching = False  # need to cache data if we want it

    if return_dictofpstrs or return_dictofvelstrs:
        def _atdct(cdict, t, thing):
            if dictkeysstr:
                cdict.update({'{0}'.format(t): thing})
            else:
                cdict.update({t: thing})
    else:
        def _atdct(cdict, t, thing):
            pass

    def _gfdct(cdict, t):
        if dictkeysstr:
            return cdict['{0}'.format(t)]
        else:
            return cdict[t]

    if stokes_flow:
        vel_nwtn_stps = 1
        vel_pcrd_stps = 0
        print('Stokes Flow!')
        comp_nonl_semexp_inig = None

    else:
        cur_linvel_point = lin_vel_point
        comp_nonl_semexp_inig = False

    newtk, norm_nwtnupd = 0, 1

    def _appbcs(vvec, ccntrlldbcvals):
        return dts.append_bcs_vec(vvec, vdim=V.dim(), invinds=dbcntinvinds,
                                  bcinds=[dbcinds, glbcntbcinds],
                                  bcvals=[dbcvals, ccntrlldbcvals])

    if treat_nonl_explct and no_data_caching:
        def _savevp(vvec, pvec, ccntrlldbcvals, cdatstr):
            pass
    else:
        def _savevp(vvec, pvec, ccntrlldbcvals, cdatstr):
            vpbc = _appbcs(vvec, ccntrlldbcvals)
            dou.save_npa(vpbc, fstring=cdatstr+'__vel')

    def _get_mats_rhs_ts(mmat=None, dt=None, var_c=None,
                         coeffmat_c=None,
                         coeffmat_n=None,
                         fv_c=None, fv_n=None,
                         umat_c=None, vmat_c=None,
                         umat_n=None, vmat_n=None,
                         mbcs_c=None, mbcs_n=None,
                         impeul=False):
        """ to be tweaked for different int schemes


        Parameters
        ---

        mbcs_c, mbcs_n: arrays
            boundary values times the corresponding part of the mass matrices
            needed for time dependent boundary conditions
        """
        solvmat = cmmat + 0.5*dt*coeffmat_n
        rhs = cmmat*var_c + 0.5*dt*(fv_n + fv_c - coeffmat_c*var_c)
        if umat_n is not None:
            umat = 0.5*dt*umat_n
            vmat = vmat_n
            # TODO: do we really need a PLUS here??'
            rhs = rhs + 0.5*dt*umat_c.dot(vmat_c.dot(var_c))
        else:
            umat, vmat = umat_n, vmat_n

        if mbcs_c is not None and mbcs_n is not None:
            rhs = rhs + mbcs_n - mbcs_c

        return solvmat, rhs, umat, vmat

# -----
# ## chap: initialization of the time integration
# -----

    v_old = iniv  # start vector for time integration in every Newtonit
    datastrdict['time'] = trange[0]
    cdatstr = get_datastring(**datastrdict)

    dictofvelstrs = {}
    _atdct(dictofvelstrs, trange[0], cdatstr + '__vel')
    p_old = inip
    cdbcvals_c = inicdbcvals
    mbcs_c = dts.condense_velmatsbybcs(M, invinds=locinvinds,
                                       dbcinds=loccntbcinds,
                                       dbcvals=inicdbcvals, get_rhs_only=True)

    _savevp(v_old, p_old, inicdbcvals, cdatstr)

    if return_dictofpstrs:
        dou.save_npa(p_old, fstring=cdatstr + '__p')
        dictofpstrs = {}
        _atdct(dictofpstrs, trange[0], cdatstr+'__p')

    if return_as_list:
        vellist = []
        vellist.append(_appbcs(v_old, inicdbcvals))

    lensect = np.int(np.floor(trange.size/nsects))
    loctrngs = []
    for k in np.arange(nsects-1):
        loctrngs.append(trange[k*lensect: (k+1)*lensect+1])
    loctrngs.append(trange[(nsects-1)*lensect:])
    if addfullsweep:
        loctrngs.append(trange)
        realiniv = np.copy(iniv)
    if nsects == 1:
        loc_nwtn_tol = vel_nwtn_tol
        addfullsweep = False
        loctrngs = [trange]
    if loc_pcrd_stps:
        vel_loc_pcrd_steps = vel_pcrd_stps

    vfile = dolfin.File(vfileprfx+'__timestep.pvd')
    pfile = dolfin.File(pfileprfx+'__timestep.pvd')

    prvoutdict.update(dict(vp=None, vc=iniv, pc=inip, t=trange[0],
                           dbcvals=[dbcvals, inicdbcvals],
                           pfile=pfile, vfile=vfile))

    dou.output_paraview(**prvoutdict)

    if lin_vel_point is None:  # do a semi-explicit integration
        from dolfin_navier_scipy.time_step_schemes import cnab

        if loccntbcinds == []:
            def applybcs(bcs_n):
                return 0., 0., 0.

        else:
            NV = J.shape[1]
            cauxvec = np.zeros((NV, 1))

            def applybcs(bcs_n):
                cauxvec[loccntbcinds, 0] = bcs_n
                return (-(A.dot(cauxvec))[locinvinds, :],
                        -(J.dot(cauxvec)),
                        (M.dot(cauxvec))[locinvinds, :])

        if fvtd is None:
            def rhsv(t):
                return cfv
        else:
            def rhsv(t):
                return cfv + fvtd(t)

        def rhsp(t):
            return fp

        def nonlvfunc(vvec):
            _, convvec, _ = \
                get_v_conv_conts(vvec=vvec, V=V,
                                 invinds=dbcntinvinds, semi_explicit=True)
            return convvec

        def getbcs(time, vvec, pvec, mode=None):
            return _comp_cntrl_bcvals(time=time, vel=vvec, p=pvec,
                                      diricontbcvals=diricontbcvals,
                                      diricontfuncs=diricontfuncs,
                                      diricontfuncmems=diricontfuncmems,
                                      mode=mode)
        if closed_loop:
            if dynamic_feedback:
                from dolfin_navier_scipy.time_step_schemes \
                    import get_heunab_lti
                dfb = dyn_fb_dict
                dyn_obs_fbk = get_heunab_lti(hb=dfb['hb'], ha=dfb['ha'],
                                             hc=dfb['hc'], inihx=dfb['inihx'],
                                             drift=dfb['drift'])

                def dynamic_rhs(t, vc=None, memory={}, mode='abtwo'):
                    cy = cv_mat.dot(vc)
                    curu, memory = dyn_obs_fbk(t, vc=cy,
                                               memory=memory, mode=mode)
                    return b_mat.dot(curu), memory
            elif static_feedback:
                pass

        else:
            dynamic_rhs = None

        expnlveldct = {}

        if return_vp_dict:
            vp_dict = {}

            def _svpplz(vvec, pvec, time=None):
                vp_dict.update({time: dict(p=pvec, v=vvec)})
                prvoutdict.update(dict(vc=vvec, pc=pvec, t=time))
                dou.output_paraview(**prvoutdict)

        elif no_data_caching and treat_nonl_explct:
            def _svpplz(vvec, pvec, time=None):
                prvoutdict.update(dict(vc=vvec, pc=pvec, t=time))
                dou.output_paraview(**prvoutdict)

        elif return_dictofvelstrs:
            def _svpplz(vvec, pvec, time=None):
                cfvstr = data_prfx + '_prs_t{0}'.format(time)
                cfpstr = data_prfx + '_vel_t{0}'.format(time)
                dou.save_npa(pvec, fstring=cfpstr)
                dou.save_npa(vvec, fstring=cfvstr)
                _atdct(expnlveldct, time, cfvstr)
                prvoutdict.update(dict(vc=vvec, pc=pvec, t=time))
                dou.output_paraview(**prvoutdict)

        v_end, p_end = cnab(trange=trange, inivel=iniv, inip=inip,
                            bcs_ini=inicdbcvals,
                            M=cmmat, A=camat, J=cj, scalep=-1.,
                            f_vdp=nonlvfunc,
                            f_tdp=rhsv, g_tdp=rhsp,
                            dynamic_rhs=dynamic_rhs,
                            getbcs=getbcs, applybcs=applybcs, appndbcs=_appbcs,
                            savevp=_svpplz)

        if treat_nonl_explct:
            if return_vp_dict:
                return vp_dict
            elif return_final_vp:
                return (v_end, p_end)
            elif return_dictofvelstrs:
                return expnlveldct
            else:
                return

        cur_linvel_point = expnlveldct
    else:
        cur_linvel_point = lin_vel_point

    for loctrng in loctrngs:

        while (newtk < vel_nwtn_stps and norm_nwtnupd > loc_nwtn_tol):
            print('solve the NSE on the interval [{0}, {1}]'.
                  format(loctrng[0], loctrng[-1]))
            v_old = iniv  # start vector for time integration in every Newtonit
            p_old = inip
            ccfv_c, ccfp_c = _cntrl_stffnss_rhs(cntrlldbcvals=cdbcvals_c,
                                                **cntrlmatrhsdict)

            if vel_pcrd_stps > 0:
                vel_pcrd_stps -= 1
                pcrd_anyone = True
                print('Picard iterations for initial value -- {0} left'.
                      format(vel_pcrd_stps))
            else:
                pcrd_anyone = False
                newtk += 1
                print('Computing Newton Iteration {0}'.format(newtk))

            try:
                if krpslvprms['krylovini'] == 'old':
                    vp_old = np.vstack([v_old, np.zeros((NP, 1))])
                elif krpslvprms['krylovini'] == 'upd':
                    vp_old = np.vstack([v_old, np.zeros((NP, 1))])
                    vp_new = vp_old
                    cts_old = loctrng[1] - loctrng[0]
            except (TypeError, KeyError):
                pass  # no inival for krylov solver required

            # ## current values_c for application of trap rule
            if stokes_flow:
                convc_mat_c = sps.csr_matrix((cnv, cnv))
                rhs_con_c = np.zeros((cnv, 1))
                rhsv_conbc_c = np.zeros((cnv, 1))
            else:
                try:
                    prev_v = dou.load_npa(_gfdct(cur_linvel_point,
                                          loctrng[0]))
                except KeyError:
                    try:
                        prev_v = dou.load_npa(_gfdct(cur_linvel_point,
                                              None))
                    except TypeError:
                        prev_v = cur_linvel_point[None]
                # prev_v = prev_v[dbcntinvinds]

                convc_mat_c, rhs_con_c, rhsv_conbc_c = \
                    get_v_conv_conts(vvec=_appbcs(v_old, cdbcvals_c), V=V,
                                     invinds=dbcntinvinds,
                                     dbcinds=[dbcinds, glbcntbcinds],
                                     dbcvals=[dbcvals, cdbcvals_c],
                                     Picard=pcrd_anyone)

            # cury = None if cv_mat is None else cv_mat.dot(v_old)
            # (fv_tmdp_cont,
            #  fv_tmdp_memory) = fv_tmdp(time=0, curvel=v_old, cury=cury,
            #                            memory=fv_tmdp_memory,
            #                            **fv_tmdp_params)

            _rhsconvc = 0. if pcrd_anyone else rhs_con_c
            fvn_c = cfv + ccfv_c + rhsv_conbc_c + _rhsconvc  # + fv_tmdp_cont

            if closed_loop:
                if static_feedback:
                    mtxtb_c = dou.load_npa(feedbackthroughdict[None]['mtxtb'])
                    w_c = dou.load_npa(feedbackthroughdict[None]['w'])
                else:
                    mtxtb_c = dou.load_npa(feedbackthroughdict[0]['mtxtb'])
                    w_c = dou.load_npa(feedbackthroughdict[0]['w'])

                fvn_c = fvn_c + b_mat * (b_mat.T * w_c)
                vmat_c = mtxtb_c.T
                try:
                    umat_c = np.array(b_mat.todense())
                except AttributeError:
                    umat_c = b_mat

            else:
                vmat_c = None
                umat_c = None

            norm_nwtnupd = 0
            if verbose:
                # define at which points of time the progress is reported
                nouts = 10  # number of output points
                locnts = loctrng.size  # TODO: trange may be a list...
                filtert = np.arange(0, locnts,
                                    np.int(np.floor(locnts/nouts)))
                loctinstances = loctrng[filtert]
                loctinstances[0] = loctrng[1]
                loctinstances = loctinstances.tolist()
                print('doing the time integration...')

# -----
# ## chap: the time stepping
# -----

            for tk, t in enumerate(loctrng[1:]):
                cts = t - loctrng[tk]
                datastrdict.update(dict(time=t))
                cdatstr = get_datastring(**datastrdict)
                try:
                    if verbose and t == loctinstances[0]:
                        curtinst = loctinstances.pop(0)
                        # print("runtime: {0} -- t: {1} -- tE: {2:f}".
                        #       format(time.clock(), curtinst, loctrng[-1]))
                        print("runtime: {0:.1f} - t/tE: {1:.2f} - t: {2:.4f}".
                              format(time.clock(), curtinst/loctrng[-1],
                                     curtinst))
                except IndexError:
                    pass  # if something goes wrong, don't stop

                # coeffs and rhs at next time instance
                if stokes_flow:
                    convc_mat_n = sps.csr_matrix((cnv, cnv))
                    rhs_con_n = np.zeros((cnv, 1))
                    rhsv_conbc_n = np.zeros((cnv, 1))
                    prev_v = v_old
                else:
                    try:
                        prev_v = dou.load_npa(_gfdct(cur_linvel_point, t))
                    except KeyError:
                        try:
                            prev_v = dou.load_npa(_gfdct(cur_linvel_point,
                                                         None))
                        except TypeError:
                            prev_v = cur_linvel_point[None]
                    prev_p = None

                cdbcvals_n = _comp_cntrl_bcvals(vel=prev_v, p=prev_p, time=t,
                                                **cntrlmatrhsdict)
                ccfv_n, ccfp_n = _cntrl_stffnss_rhs(cntrlldbcvals=cdbcvals_n,
                                                    **cntrlmatrhsdict)
                mbcs_n = dts.condense_velmatsbybcs(M, invinds=locinvinds,
                                                   dbcinds=loccntbcinds,
                                                   dbcvals=cdbcvals_n,
                                                   get_rhs_only=True)

                convc_mat_n, rhs_con_n, rhsv_conbc_n = \
                    get_v_conv_conts(vvec=prev_v, V=V,
                                     invinds=dbcntinvinds,
                                     dbcinds=[dbcinds, glbcntbcinds],
                                     dbcvals=[dbcvals, cdbcvals_n],
                                     Picard=pcrd_anyone)

                # cury = None if cv_mat is None else cv_mat.dot(prev_v)
                # (fv_tmdp_cont,
                #  fv_tmdp_memory) = fv_tmdp(time=t,
                #                            curvel=prev_v,
                #                            cury=cury,
                #                            memory=fv_tmdp_memory,
                #                            **fv_tmdp_params)

                _rhsconvn = 0. if pcrd_anyone else rhs_con_n
                fvn_n = cfv + ccfv_n + rhsv_conbc_n + _rhsconvn  # + fv_tmdp

                if closed_loop:
                    if static_feedback:
                        mtxtb_n = dou.\
                            load_npa(feedbackthroughdict[None]['mtxtb'])
                        w_n = dou.load_npa(feedbackthroughdict[None]['w'])
                        # fb = np.dot(tb_mat*mtxtb_n.T, v_old)
                        # print '\nnorm of feedback: ', np.linalg.norm(fb)
                        # print '\nnorm of v_old: ', np.linalg.norm(v_old)
                    else:
                        mtxtb_n = dou.load_npa(feedbackthroughdict[t]['mtxtb'])
                        w_n = dou.load_npa(feedbackthroughdict[t]['w'])

                    fvn_n = fvn_n + b_mat * (b_mat.T * w_n)
                    vmat_n = mtxtb_n.T
                    try:
                        umat_n = np.array(b_mat.todense())
                    except AttributeError:
                        umat_n = b_mat

                else:
                    vmat_n = None
                    umat_n = None

                (solvmat, rhsv, umat,
                 vmat) = _get_mats_rhs_ts(mmat=cmmat, dt=cts, var_c=v_old,
                                          coeffmat_c=camat + convc_mat_c,
                                          coeffmat_n=camat + convc_mat_n,
                                          fv_c=fvn_c, fv_n=fvn_n,
                                          umat_c=umat_c, vmat_c=vmat_c,
                                          umat_n=umat_n, vmat_n=vmat_n,
                                          mbcs_c=mbcs_c, mbcs_n=mbcs_n)

                try:
                    if krpslvprms['krylovini'] == 'old':
                        krpslvprms['x0'] = vp_old
                    elif krpslvprms['krylovini'] == 'upd':
                        vp_oldold = vp_old
                        vp_old = vp_new
                        krpslvprms['x0'] = vp_old + \
                            cts*(vp_old - vp_oldold)/cts_old
                        cts_old = cts
                except (TypeError, KeyError):
                    pass  # no inival for krylov solver required

                vp_new = lau.solve_sadpnt_smw(amat=solvmat,
                                              jmat=cj, jmatT=cjt,
                                              rhsv=rhsv,
                                              rhsp=cfp+ccfp_n,
                                              krylov=krylov,
                                              krpslvprms=krpslvprms,
                                              krplsprms=krplsprms,
                                              umat=umat, vmat=vmat)

                # print('v_old : {0} ({1})'.format(np.linalg.norm(v_old),
                #                                  v_old.size))
                v_old = vp_new[:cnv, ]
                # print('v_new : {0} ({1})'.format(np.linalg.norm(v_old),
                #                                  v_old.size))
                # print('v_prv : {0} ({1})'.format(np.linalg.norm(prev_v),
                #                                  prev_v.size))

# -----
# ## chap: preparing for the next time step
# -----
                umat_c, vmat_c = umat_n, vmat_n
                cdbcvals_c = cdbcvals_n
                mbcs_c = mbcs_n

                convc_mat_c, rhs_con_c, rhsv_conbc_c = \
                    get_v_conv_conts(vvec=_appbcs(v_old, cdbcvals_n), V=V,
                                     invinds=dbcntinvinds,
                                     dbcinds=[dbcinds, glbcntbcinds],
                                     dbcvals=[dbcvals, cdbcvals_n],
                                     Picard=pcrd_anyone)

                _rhsconvc = 0. if pcrd_anyone else rhs_con_c
                fvn_c = (fvn_n - _rhsconvn - rhsv_conbc_n
                         + rhsv_conbc_c + _rhsconvc)

                _savevp(v_old, p_old, cdbcvals_n, cdatstr)
                _atdct(dictofvelstrs, t, cdatstr + '__vel')
                p_old = -1/cts*vp_new[cnv:, ]
                # p was flipped and scaled for symmetry
                if return_dictofpstrs:
                    dou.save_npa(p_old, fstring=cdatstr + '__p')
                    _atdct(dictofpstrs, t, cdatstr + '__p')

                if return_as_list:
                    vellist.append(_appbcs(v_old, cdbcvals_n))

                # integrate the Newton error
                if stokes_flow or treat_nonl_explct:
                    norm_nwtnupd = None
                elif comp_nonl_semexp_inig:
                    norm_nwtnupd = 1.
                else:
                    if len(prev_v) > len(locinvinds):
                        prev_v = prev_v[dbcntinvinds, :]
                    addtonwtnupd = cts * m_innerproduct(cmmat, v_old - prev_v)
                    norm_nwtnupd += np.float(addtonwtnupd.flatten()[0])

                if newtk == vel_nwtn_stps or norm_nwtnupd < loc_nwtn_tol:
                    # paraviewoutput in the (probably) last newton sweep
                    prvoutdict.update(dict(vc=v_old, pc=p_old, t=t,
                                           dbcvals=[dbcvals, cdbcvals_c]))
                    dou.output_paraview(**prvoutdict)

            dou.save_npa(norm_nwtnupd, cdatstr + '__norm_nwtnupd')
            print('\nnorm of current Newton update: {}'.format(norm_nwtnupd))
            # print('\nsaved `norm_nwtnupd(={0})'.format(norm_nwtnupd) +
            #       ' to ' + cdatstr)

            cur_linvel_point = dictofvelstrs

        iniv = v_old  # overwrite iniv as the starting value
        inip = p_old  # > for the next time section

        if addfullsweep and loctrng is loctrngs[-2]:
            comp_nonl_semexp_inig = False
            iniv = realiniv
            loc_nwtn_tol = vel_nwtn_tol
        elif loc_pcrd_stps:
            vel_pcrd_stps = vel_loc_pcrd_steps

        norm_nwtnupd = 1.
        newtk = 0

    if return_final_vp:
        return (_appbcs(v_old, cdbcvals_n), p_old)
    elif return_dictofvelstrs:
        if return_dictofpstrs:
            return dictofvelstrs, dictofpstrs
        else:
            return dictofvelstrs
    elif return_as_list:
        return vellist
    else:
        return


def get_pfromv(v=None, V=None, M=None, A=None, J=None, fv=None, fp=None,
               decouplevp=False, solve_M=None, symmetric=False,
               cgtol=1e-8,
               diribcs=None, dbcinds=None, dbcvals=None, invinds=None,
               **kwargs):
    """ for a velocity `v`, get the corresponding `p`

    Notes
    -----
    Formula is only valid for constant rhs in the continuity equation
    """

    import sadptprj_riclyap_adi.lin_alg_utils as lau

    _, rhs_con, _ = get_v_conv_conts(vvec=v, V=V, invinds=invinds,
                                     dbcinds=dbcinds, dbcvals=dbcvals)

    if decouplevp and symmetric:
        vp = lau.solve_sadpnt_smw(jmat=J, jmatT=J.T,
                                  decouplevp=decouplevp, solve_A=solve_M,
                                  symmetric=symmetric, cgtol=1e-8,
                                  rhsv=-A*v-rhs_con+fv)
        return -vp[J.shape[1]:, :]
    else:
        vp = lau.solve_sadpnt_smw(amat=M, jmat=J, jmatT=J.T,
                                  decouplevp=decouplevp, solve_A=solve_M,
                                  symmetric=symmetric, cgtol=1e-8,
                                  rhsv=-A*v-rhs_con+fv)
        return -vp[J.shape[1]:, :]
