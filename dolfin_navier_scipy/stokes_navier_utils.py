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
    sestr = '_semiexpl' if semiexpl else ''
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


def get_v_conv_conts(prev_v=None, V=None, invinds=None, diribcs=None,
                     dbcvals=None, dbcinds=None,
                     Picard=False, retparts=False, zerodiribcs=False):
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
    prev_v : (N,1) ndarray
        convection velocity
    V : dolfin.VectorFunctionSpace
        FEM space of the velocity
    invinds : (N,) ndarray or list
        indices of the inner nodes
    diribcs : list
        of dolfin Dirichlet boundary conditons
    Picard : Boolean
        whether Picard linearization is applied, defaults to `False`
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

    """

    N1, N2, rhs_con = dts.get_convmats(u0_vec=prev_v, V=V, invinds=invinds,
                                       dbcinds=dbcinds, dbcvals=dbcvals,
                                       diribcs=diribcs)

    if zerodiribcs:
        def _cndnsmts(mat, diribcs, **kw):
            return mat[invinds, :][:, invinds], np.zeros((invinds.size, 1))
    else:
        _cndnsmts = dts.condense_velmatsbybcs

    if Picard:
        convc_mat, rhsv_conbc = _cndnsmts(N1, velbcs=diribcs,
                                          dbcinds=dbcinds, dbcvals=dbcvals)
        # return convc_mat, rhs_con[invinds, ], rhsv_conbc
        return convc_mat, None, rhsv_conbc

    elif retparts:
        picrd_convc_mat, picrd_rhsv_conbc = _cndnsmts(N1, velbcs=diribcs,
                                                      dbcinds=dbcinds,
                                                      dbcvals=dbcvals)
        anti_picrd_convc_mat, anti_picrd_rhsv_conbc = \
            _cndnsmts(N2, velbcs=diribcs, dbcinds=dbcinds, dbcvals=dbcvals)
        return ((picrd_convc_mat, anti_picrd_convc_mat),
                rhs_con[invinds, ],
                (picrd_rhsv_conbc, anti_picrd_rhsv_conbc))

    else:
        convc_mat, rhsv_conbc = _cndnsmts(N1+N2, velbcs=diribcs,
                                          dbcinds=dbcinds, dbcvals=dbcvals)
        return convc_mat, rhs_con[invinds, ], rhsv_conbc


def m_innerproduct(M, v1, v2=None):
    """ inner product with a spd sparse matrix

    """
    if v2 is None:
        v2 = v1  # in most cases, we want to compute the norm

    return np.dot(v1.T, M*v2)


def _unroll_cntrl_dbcs(diricontbcvals, diricontfuncs, time=None, vel=None):
    cntrlldbcvals = []
    try:
        for k, cdbbcv in enumerate(diricontbcvals):
            ccntrlfunc = diricontfuncs[k]
            cntrlval = ccntrlfunc(time, vel)
            ccntrlldbcvals = [cntrlval*bcvl for bcvl in cdbbcv]
            cntrlldbcvals.extend(ccntrlldbcvals)
    except TypeError:
        pass
    return cntrlldbcvals


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
                          diricontfuncs=None,
                          return_vp=False, ppin=-1,
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
        which dof of `p` is used to pin the pressure, defaults to `-1`
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
    vel_[p]k : (N, 1) ndarray
        the velocity/[pressure] vector. Pressure only if `return_vp`.
    norm_nwtnupd_list : list, on demand
        list of the newton upd errors
    """

    import sadptprj_riclyap_adi.lin_alg_utils as lau

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if JT is None:
        JT = J.T

    NV = J.shape[1]

#
# Compute or load the uncontrolled steady state Navier-Stokes solution
#

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
                                     invinds=invinds, diribcs=diribcs)
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
    if vel_start_nwtn is None:
        loccntbcinds, cntrlldbcvals, glbcntbcinds = [], [], []
        if diricontbcinds is None or diricontbcinds == []:
            cmmat, camat, cj, cjt, cfv, cfp = M, A, J, JT, fv, fp
            cnv = NV
            dbcntinvinds = invinds
        else:
            def _localizecdbinds(cdbinds):
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

            for k, cdbidbv in enumerate(diricontbcinds):
                ccntrlfunc = diricontfuncs[k]

                # no time at steady state, no starting value
                cntrlval = ccntrlfunc(None, None)

                localbcinds = (_localizecdbinds(cdbidbv)).tolist()
                loccntbcinds.extend(localbcinds)  # adding the boundary inds
                glbcntbcinds.extend(cdbidbv)
                ccntrlldbcvals = [cntrlval*bcvl for bcvl in diricontbcvals[k]]
                # adding the scaled boundary values
                cntrlldbcvals.extend(ccntrlldbcvals)

            dbcntinvinds = np.setdiff1d(invinds, glbcntbcinds).astype(np.int32)
            matdict = dict(M=M, A=A, J=J, JT=JT, MP=None)
            cmmat, camat, cjt, cj, _, cfv, cfp, _ = dts.\
                condense_sysmatsbybcs(matdict, dbcinds=loccntbcinds,
                                      dbcvals=cntrlldbcvals, mergerhs=True,
                                      rhsdict=dict(fv=fv, fp=fp),
                                      ret_unrolled=True)
            cnv = cmmat.shape[0]

        vp_stokes = lau.solve_sadpnt_smw(amat=camat, jmat=cj, jmatT=cjt,
                                         rhsv=cfv, rhsp=cfp)
        vp_stokes[cnv:] = -vp_stokes[cnv:]
        # pressure was flipped for symmetry

        # save the data
        cdatstr = get_datastring(**datastrdict)

        if save_data:
            dou.save_npa(vp_stokes[:cnv, ], fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_stokes, dbcinds=[dbcinds, glbcntbcinds],
                               dbcvals=[dbcvals, cntrlldbcvals],
                               invinds=dbcntinvinds))
        dou.output_paraview(**prvoutdict)

        # Stokes solution as starting value
        vp_k = vp_stokes
        vel_k = vp_stokes[:cnv, ]

    else:
        vel_k = vel_start_nwtn

    matdict = dict(M=M, A=A, J=J, JT=JT, MP=None)
    rhsdict = dict(fv=fv, fp=fp)
    cndnsmtsdct = dict(dbcinds=loccntbcinds, mergerhs=True,
                       ret_unrolled=True)

    # Picard iterations for a good starting value for Newton
    for k in range(vel_pcrd_stps):

        cntrlldbcvals = _unroll_cntrl_dbcs(diricontbcvals, diricontfuncs,
                                           time=None, vel=vel_k)
        (convc_mat,
         rhs_con, rhsv_conbc) = \
            get_v_conv_conts(prev_v=vel_k, V=V, diribcs=diribcs,
                             invinds=dbcntinvinds,
                             dbcinds=[dbcinds, glbcntbcinds],
                             dbcvals=[dbcvals, cntrlldbcvals], Picard=True)

        _, _, _, _, _, cfv, cfp, _ = dts.\
            condense_sysmatsbybcs(matdict, dbcvals=cntrlldbcvals,
                                  rhsdict=rhsdict, **cndnsmtsdct)

        vp_k = lau.solve_sadpnt_smw(amat=camat+convc_mat, jmat=cj, jmatT=cjt,
                                    rhsv=cfv+rhsv_conbc, rhsp=cfp)
        # vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
        #                             rhsv=fv+rhsv_conbc,
        #                             rhsp=fp)

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

        cntrlldbcvals = _unroll_cntrl_dbcs(diricontbcvals, diricontfuncs,
                                           time=None, vel=vel_k)
        _, _, _, _, _, cfv, cfp, _ = dts.\
            condense_sysmatsbybcs(matdict, dbcvals=cntrlldbcvals,
                                  rhsdict=rhsdict, **cndnsmtsdct)
        (convc_mat, rhs_con, rhsv_conbc) = \
            get_v_conv_conts(prev_v=vel_k, V=V, diribcs=diribcs,
                             invinds=dbcntinvinds,
                             dbcinds=[dbcinds, glbcntbcinds],
                             dbcvals=[dbcvals, cntrlldbcvals])

        vp_k = lau.solve_sadpnt_smw(amat=camat+convc_mat, jmat=cj, jmatT=cjt,
                                    rhsv=cfv+rhs_con+rhsv_conbc,
                                    rhsp=cfp)

        norm_nwtnupd = np.sqrt(m_innerproduct(cmmat, vel_k - vp_k[:cnv, :]))[0]
        vel_k = vp_k[:cnv, ]
        vp_k[cnv:] = -vp_k[cnv:]
        # pressure was flipped for symmetry
        if verbose:
            print('Steady State NSE: Newton iteration: {0}'.format(vel_newtk) +
                  '-- norm of update: {0}'.format(norm_nwtnupd))

        if save_data:
            dou.save_npa(vel_k, fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k, dbcvals=[dbcvals, cntrlldbcvals]))
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

    prvoutdict.update(dict(vp=vp_k, dbcvals=[dbcvals, cntrlldbcvals]))
    dou.output_paraview(**prvoutdict)

    # savetomatlab = True
    # if savetomatlab:
    #     export_mats_to_matlab(E=None, A=None, matfname='matexport')

    vwc = _attach_cntbcvals(vel_k.flatten(), globbcinvinds=dbcntinvinds,
                            globbcinds=glbcntbcinds, dbcvals=cntrlldbcvals,
                            invinds=invinds, NV=V.dim())
    if return_vp:
        retthing = (vwc.reshape((NV, 1)), vp_k[cnv:, :])
    else:
        retthing = vwc.reshape((NV, 1))

    if return_nwtnupd_norms:
        return retthing, norm_nwtnupd_list
    else:
        return retthing


def solve_nse(A=None, M=None, J=None, JT=None,
              fv=None, fp=None,
              fvc=None, fpc=None,  # TODO: this is to catch deprecated calls
              fv_tmdp=None, fv_tmdp_params={},
              fv_tmdp_memory=None,
              iniv=None, lin_vel_point=None,
              stokes_flow=False,
              trange=None,
              t0=None, tE=None, Nts=None,
              V=None, Q=None, invinds=None, diribcs=None,
              dbcinds=None, dbcvals=None,
              output_includes_bcs=False,
              N=None, nu=None,
              ppin=-1,
              closed_loop=False, static_feedback=False,
              feedbackthroughdict=None,
              return_vp=False,
              tb_mat=None, cv_mat=None,
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
              comp_nonl_semexp=False,
              return_as_list=False,
              verbose=True,
              start_ssstokes=False,
              **kw):
    """
    solution of the time-dependent nonlinear Navier-Stokes equation

    .. math::
        M\\dot v + Av + N(v)v + J^Tp = f \n
        Jv =g

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
    fv_tmdp : callable f(t, v, dict), optional
        time-dependent part of the right-hand side, set to zero if None
    fv_tmdp_params : dictionary, optional
        dictionary of parameters to be passed to `fv_tmdp`, defaults to `{}`
    fv_tmdp_memory : dictionary, optional
        memory of the function
    output_includes_bcs : boolean, optional
        whether append the boundary nodes to the computed and stored \
        velocities, defaults to `False`
    krylov : {None, 'gmres'}, optional
        whether or not to use an iterative solver, defaults to `None`
    krpslvprms : dictionary, optional
        to specify parameters of the linear solver for use in Krypy, e.g.,

          * initial guess
          * tolerance
          * number of iterations

        defaults to `None`
    krplsprms : dictionary, optional
        parameters to define the linear system like

          * preconditioner

    ppin : {int, None}, optional
        which dof of `p` is used to pin the pressure, defaults to `-1`
    stokes_flow : boolean, optional
        whether to consider the Stokes linearization, defaults to `False`
    start_ssstokes : boolean, optional
        for your convenience, compute and use the steady state stokes solution
        as initial value, defaults to `False`
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

    if fvc is not None or fpc is not None:  # TODO: this is for catching calls
        raise UserWarning('deprecated use of `rhsd_vfrc`, use only `fv`, `fp`')

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if paraviewoutput:
        prvoutdict = dict(V=V, Q=Q,
                          invinds=invinds, diribcs=diribcs, ppin=ppin,
                          vp=None, t=None,
                          tfilter=plttrange, writeoutput=True)
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' here

    if trange is None:
        trange = np.linspace(t0, tE, Nts+1)

    if comp_nonl_semexp and lin_vel_point is not None:
        raise UserWarning('I am not sure what you want! ' +
                          'set either `lin_vel_point=None` ' +
                          'or `comp_nonl_semexp=False`! \n' +
                          'as it is, I will compute a linear case')

    if return_dictofpstrs:
        gpfvd = dict(V=V, M=M, A=A, J=J,
                     fv=fv, fp=fp,
                     dbcinds=dbcinds, dbcvals=dbcvals,
                     diribcs=diribcs, invinds=invinds)

    NV, NP = A.shape[0], J.shape[0]
    fv = np.zeros((NV, 1)) if fv is None else fv
    fp = np.zeros((NP, 1)) if fp is None else fp

    if fv_tmdp is None:
        def fv_tmdp(time=None, curvel=None, **kw):
            return np.zeros((NV, 1)), None

    if iniv is None:
        if start_ssstokes:
            # Stokes solution as starting value
            vp_stokes =\
                lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                     rhsv=fv,  # + fv_tmdp_cont,
                                     krylov=krylov, krpslvprms=krpslvprms,
                                     krplsprms=krplsprms, rhsp=fp)
            iniv = vp_stokes[:NV]
        else:
            raise ValueError('No initial value given')

    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=trange.size-1, data_prfx=data_prfx,
                       semiexpl=comp_nonl_semexp)

    if return_as_list:
        clearprvdata = True  # we want the results at hand
    if clearprvdata:
        datastrdict['time'] = '*'
        cdatstr = get_datastring(**datastrdict)
        for fname in glob.glob(cdatstr + '__vel*'):
            os.remove(fname)
        for fname in glob.glob(cdatstr + '__p*'):
            os.remove(fname)

    def _atdct(cdict, t, thing):
        if dictkeysstr:
            cdict.update({'{0}'.format(t): thing})
        else:
            cdict.update({t: thing})

    def _gfdct(cdict, t):
        if dictkeysstr:
            return cdict['{0}'.format(t)]
        else:
            return cdict[t]

    if stokes_flow:
        vel_nwtn_stps = 1
        vel_pcrd_stps = 0
        print('Stokes Flow!')
    elif lin_vel_point is None:
        comp_nonl_semexp_inig = True
        if not comp_nonl_semexp:
            print(('No linearization point given - explicit' +
                  ' treatment of the nonlinearity in the first Iteration'))
    else:
        cur_linvel_point = lin_vel_point
        comp_nonl_semexp_inig = False

    newtk, norm_nwtnupd = 0, 1

    # check for previously computed velocities
    if useolddata and lin_vel_point is None and not stokes_flow:
        try:
            datastrdict.update(dict(time=trange[-1]))
            cdatstr = get_datastring(**datastrdict)

            norm_nwtnupd = (dou.load_npa(cdatstr + '__norm_nwtnupd')).flatten()
            try:
                if norm_nwtnupd[0] is None:
                    norm_nwtnupd = 1.
            except IndexError:
                norm_nwtnupd = 1.

            dou.load_npa(cdatstr + '__vel')

            print('found vel files')
            print('norm of last Nwtn update: {0}'.format(norm_nwtnupd))
            print('... loaded from ' + cdatstr)

            if norm_nwtnupd < vel_nwtn_tol and not return_dictofvelstrs:
                return
            elif norm_nwtnupd < vel_nwtn_tol or comp_nonl_semexp:
                # looks like converged -- check if all values are there
                # t0:
                datastrdict.update(dict(time=trange[0]))
                cdatstr = get_datastring(**datastrdict)
                dictofvelstrs = {}
                _atdct(dictofvelstrs, trange[0], cdatstr + '__vel')
                if return_dictofpstrs:
                    dictofpstrs = {}

                for t in trange:
                    datastrdict.update(dict(time=t))
                    cdatstr = get_datastring(**datastrdict)
                    # test if the vels are there
                    v_old = dou.load_npa(cdatstr + '__vel')
                    # update the dict
                    _atdct(dictofvelstrs, t, cdatstr + '__vel')
                    if return_dictofpstrs:
                        try:
                            p_old = dou.load_npa(cdatstr + '__p')
                            _atdct(dictofpstrs, t, cdatstr + '__p')
                        except:
                            p_old = get_pfromv(v=v_old, **gpfvd)
                            dou.save_npa(p_old, fstring=cdatstr + '__p')
                            _atdct(dictofpstrs, t, cdatstr + '__p')

                if return_dictofpstrs:
                    return dictofvelstrs, dictofpstrs
                else:
                    return dictofvelstrs

            # comp_nonl_semexp = False

        except IOError:
            norm_nwtnupd = 2
            print('no old velocity data found')

    def _append_bcs_ornot(vvec):
        if output_includes_bcs:  # make the switch here for better readibility
            vwbcs = dts.append_bcs_vec(vvec, vdim=V.dim(),
                                       invinds=invinds, diribcs=diribcs)
            return vwbcs
        else:
            return vvec

    def _get_mats_rhs_ts(mmat=None, dt=None, var_c=None,
                         coeffmat_c=None,
                         coeffmat_n=None,
                         fv_c=None, fv_n=None,
                         umat_c=None, vmat_c=None,
                         umat_n=None, vmat_n=None,
                         impeul=False):
        """ to be tweaked for different int schemes

        """
        solvmat = M + 0.5*dt*coeffmat_n
        rhs = M*var_c + 0.5*dt*(fv_n + fv_c - coeffmat_c*var_c)
        if umat_n is not None:
            matvec = lau.mm_dnssps
            umat = 0.5*dt*umat_n
            vmat = vmat_n
            # TODO: do we really need a PLUS here??'
            rhs = rhs + 0.5*dt*matvec(umat_c, matvec(vmat_c, var_c))
        else:
            umat, vmat = umat_n, vmat_n

        return solvmat, rhs, umat, vmat

    v_old = iniv  # start vector for time integration in every Newtonit
    datastrdict['time'] = trange[0]
    cdatstr = get_datastring(**datastrdict)

    dou.save_npa(_append_bcs_ornot(v_old), fstring=cdatstr + '__vel')
    dictofvelstrs = {}
    _atdct(dictofvelstrs, trange[0], cdatstr + '__vel')
    if return_dictofpstrs:
        p_old = get_pfromv(v=v_old, **gpfvd)
        dou.save_npa(p_old, fstring=cdatstr + '__p')
        dictofpstrs = {}
        _atdct(dictofpstrs, trange[0], cdatstr+'__p')
    else:
        p_old = None

    if return_as_list:
        vellist = []
        vellist.append(_append_bcs_ornot(v_old))

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

    for loctrng in loctrngs:
        while (newtk < vel_nwtn_stps and norm_nwtnupd > loc_nwtn_tol):
            print('solve the NSE on the interval [{0}, {1}]'.
                  format(loctrng[0], loctrng[-1]))
            if stokes_flow:
                pcrd_anyone = False
                newtk = vel_nwtn_stps
            elif comp_nonl_semexp_inig and not comp_nonl_semexp:
                pcrd_anyone = False
                print('explicit treatment of nonl. for initial guess')
            elif vel_pcrd_stps > 0 and not comp_nonl_semexp:
                vel_pcrd_stps -= 1
                pcrd_anyone = True
                print('Picard iterations for initial value -- {0} left'.
                      format(vel_pcrd_stps))
            elif comp_nonl_semexp:
                pcrd_anyone = False
                newtk = vel_nwtn_stps
                print('No Newton iterations - explicit treatment ' +
                      'of the nonlinearity')
            else:
                pcrd_anyone = False
                newtk += 1
                print('Computing Newton Iteration {0}'.format(newtk))

            v_old = iniv  # start vector for time integration in every Newtonit
            try:
                if krpslvprms['krylovini'] == 'old':
                    vp_old = np.vstack([v_old, np.zeros((NP, 1))])
                elif krpslvprms['krylovini'] == 'upd':
                    vp_old = np.vstack([v_old, np.zeros((NP, 1))])
                    vp_new = vp_old
                    cts_old = loctrng[1] - loctrng[0]
            except (TypeError, KeyError):
                pass  # no inival for krylov solver required

            vfile = dolfin.File(vfileprfx+'__timestep.pvd')
            pfile = dolfin.File(pfileprfx+'__timestep.pvd')
            prvoutdict.update(dict(vp=None, vc=iniv, pc=p_old, t=loctrng[0],
                                   dbcinds=dbcinds, dbcvals=dbcvals,
                                   pfile=pfile, vfile=vfile))
            dou.output_paraview(**prvoutdict)

            # ## current values_c for application of trap rule
            if stokes_flow:
                convc_mat_c = sps.csr_matrix((NV, NV))
                rhs_con_c, rhsv_conbc_c = np.zeros((NV, 1)), np.zeros((NV, 1))
            else:
                if comp_nonl_semexp or comp_nonl_semexp_inig:
                    prev_v = v_old
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

                convc_mat_c, rhs_con_c, rhsv_conbc_c = \
                    get_v_conv_conts(prev_v=iniv, invinds=invinds,
                                     dbcinds=dbcinds, dbcvals=dbcvals,
                                     V=V, diribcs=diribcs, Picard=pcrd_anyone)

            cury = None if cv_mat is None else cv_mat.dot(v_old)
            (fv_tmdp_cont,
             fv_tmdp_memory) = fv_tmdp(time=0,
                                       curvel=v_old,
                                       cury=cury,
                                       memory=fv_tmdp_memory,
                                       **fv_tmdp_params)

            _rhsconvc = 0. if pcrd_anyone else rhs_con_c
            fvn_c = fv + rhsv_conbc_c + _rhsconvc + fv_tmdp_cont

            if closed_loop:
                if static_feedback:
                    mtxtb_c = dou.load_npa(feedbackthroughdict[None]['mtxtb'])
                    w_c = dou.load_npa(feedbackthroughdict[None]['w'])
                else:
                    mtxtb_c = dou.load_npa(feedbackthroughdict[0]['mtxtb'])
                    w_c = dou.load_npa(feedbackthroughdict[0]['w'])

                fvn_c = fvn_c + tb_mat * (tb_mat.T * w_c)
                vmat_c = mtxtb_c.T
                try:
                    umat_c = np.array(tb_mat.todense())
                except AttributeError:
                    umat_c = tb_mat

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
                    convc_mat_n = sps.csr_matrix((NV, NV))
                    rhs_con_n = np.zeros((NV, 1))
                    rhsv_conbc_n = np.zeros((NV, 1))
                else:
                    if comp_nonl_semexp or comp_nonl_semexp_inig:
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
                    convc_mat_n, rhs_con_n, rhsv_conbc_n = \
                        get_v_conv_conts(prev_v=prev_v, invinds=invinds, V=V,
                                         dbcinds=dbcinds, dbcvals=dbcvals,
                                         diribcs=diribcs, Picard=pcrd_anyone)

                cury = None if cv_mat is None else cv_mat.dot(v_old)
                (fv_tmdp_cont,
                 fv_tmdp_memory) = fv_tmdp(time=t,
                                           curvel=v_old,
                                           cury=cury,
                                           memory=fv_tmdp_memory,
                                           **fv_tmdp_params)

                _rhsconvn = 0. if pcrd_anyone else rhs_con_n
                fvn_n = fv + rhsv_conbc_n + _rhsconvn + fv_tmdp_cont

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

                    fvn_n = fvn_n + tb_mat * (tb_mat.T * w_n)
                    vmat_n = mtxtb_n.T
                    try:
                        umat_n = np.array(tb_mat.todense())
                    except AttributeError:
                        umat_n = tb_mat

                else:
                    vmat_n = None
                    umat_n = None

                (solvmat, rhsv, umat,
                 vmat) = _get_mats_rhs_ts(mmat=M, dt=cts, var_c=v_old,
                                          coeffmat_c=A + convc_mat_c,
                                          coeffmat_n=A + convc_mat_n,
                                          fv_c=fvn_c, fv_n=fvn_n,
                                          umat_c=umat_c, vmat_c=vmat_c,
                                          umat_n=umat_n, vmat_n=vmat_n)

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
                                              jmat=J, jmatT=JT,
                                              rhsv=rhsv,
                                              rhsp=fp,
                                              krylov=krylov,
                                              krpslvprms=krpslvprms,
                                              krplsprms=krplsprms,
                                              umat=umat, vmat=vmat)

                v_old = vp_new[:NV, ]
                (umat_c, vmat_c, fvn_c,
                    convc_mat_c) = umat_n, vmat_n, fvn_n, convc_mat_n

                dou.save_npa(_append_bcs_ornot(v_old),
                             fstring=cdatstr + '__vel')
                _atdct(dictofvelstrs, t, cdatstr + '__vel')
                p_new = -1/cts*vp_new[NV:, ]
                # p was flipped and scaled for symmetry
                if return_dictofpstrs:
                    dou.save_npa(p_new, fstring=cdatstr + '__p')
                    _atdct(dictofpstrs, t, cdatstr + '__p')

                if return_as_list:
                    vellist.append(_append_bcs_ornot(v_old))

                prvoutdict.update(dict(vc=v_old, pc=p_new, t=t))
                dou.output_paraview(**prvoutdict)

                # integrate the Newton error
                if stokes_flow or comp_nonl_semexp:
                    norm_nwtnupd = None
                elif comp_nonl_semexp_inig:
                    norm_nwtnupd = 1.

                else:
                    if len(prev_v) > len(invinds):
                        prev_v = prev_v[invinds, :]
                    addtonwtnupd = cts * m_innerproduct(M, v_old - prev_v)
                    norm_nwtnupd += np.float(addtonwtnupd.flatten()[0])

            dou.save_npa(norm_nwtnupd, cdatstr + '__norm_nwtnupd')
            print('\nnorm of current Newton update: {}'.format(norm_nwtnupd))
            # print('\nsaved `norm_nwtnupd(={0})'.format(norm_nwtnupd) +
            #       ' to ' + cdatstr)
            comp_nonl_semexp = False
            comp_nonl_semexp_inig = False

            cur_linvel_point = dictofvelstrs

        iniv = v_old
        if not comp_nonl_semexp:
            comp_nonl_semexp_inig = True
        if addfullsweep and loctrng is loctrngs[-2]:
            comp_nonl_semexp_inig = False
            iniv = realiniv
            loc_nwtn_tol = vel_nwtn_tol
        elif loc_pcrd_stps:
            vel_pcrd_stps = vel_loc_pcrd_steps

        norm_nwtnupd = 1.
        newtk = 0

    if return_dictofvelstrs:
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

    _, rhs_con, _ = get_v_conv_conts(prev_v=v, V=V, invinds=invinds,
                                     dbcinds=dbcinds, dbcvals=dbcvals,
                                     diribcs=diribcs)

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
