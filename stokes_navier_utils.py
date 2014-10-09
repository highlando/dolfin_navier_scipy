import numpy as np
import os
import glob
import sys
import copy
import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def get_datastr_snu(time=None, meshp=None, nu=None, Nts=None, data_prfx=''):

    return (data_prfx +
            'time{0}_nu{1}_mesh{2}_Nts{3}').format(time, nu, meshp, Nts)


def get_v_conv_conts(prev_v=None, V=None, invinds=None, diribcs=None,
                     Picard=False):
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
    Picard : boolean
        whether Picard linearization is applied, defaults to `False`

    Returns
    -------
    convc_mat : (N,N) sparse matrix
        representing the linearized convection at the inner nodes
    rhs_con : (N,1) array
        representing :math:`(u_0 \\cdot \\nabla )u_0` at the inner nodes
    rhsv_conbc : (N,1) ndarray
        representing the boundary conditions

    """

    N1, N2, rhs_con = dts.get_convmats(u0_vec=prev_v,
                                       V=V,
                                       invinds=invinds,
                                       diribcs=diribcs)
    if Picard:
        convc_mat, rhsv_conbc = \
            dts.condense_velmatsbybcs(N1, diribcs)
        return convc_mat, 0*rhs_con[invinds, ], rhsv_conbc

    else:
        convc_mat, rhsv_conbc = \
            dts.condense_velmatsbybcs(N1 + N2, diribcs)
        return convc_mat, rhs_con[invinds, ], rhsv_conbc


def m_innerproduct(M, v1, v2=None):
    """ inner product with a spd sparse matrix

    """
    if v2 is None:
        v2 = v1  # in most cases, we want to compute the norm

    return np.dot(v1.T, M*v2)


def solve_steadystate_nse(A=None, J=None, JT=None, M=None,
                          fvc=None, fpr=None,
                          fv_stbc=None, fp_stbc=None,
                          V=None, Q=None, invinds=None, diribcs=None,
                          N=None, nu=None,
                          vel_pcrd_stps=100, vel_pcrd_tol=1e-4,
                          vel_nwtn_stps=20, vel_nwtn_tol=5e-15,
                          clearprvdata=False,
                          vel_start_nwtn=None,
                          get_datastring=None,
                          data_prfx='',
                          paraviewoutput=False,
                          save_intermediate_steps=False,
                          vfileprfx='', pfileprfx='',
                          **kw):

    """
    Solution of the steady state nonlinear NSE Problem

    using Newton's scheme. If no starting value is provide, the iteration
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
    fvc, fpr : (N,1), (M,1) ndarrays
        right hand sides restricted via removing the boundary nodes in the
        momentum and the pressure freedom in the continuity equation
    fv_stbc, fp_stbc : (N,1), (M,1) ndarrays
        contributions to the right hand side by the Dirichlet boundary
        conditions in the Stokes equations.
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

    """

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if JT is None:
        JT = J.T

#
# Compute or load the uncontrolled steady state Navier-Stokes solution
#

    norm_nwtnupd_list = []
    vel_newtk, norm_nwtnupd = 0, 1
    # a dict to be passed to the get_datastring function
    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=None, data_prfx=data_prfx)

    if clearprvdata:
        cdatstr = get_datastring(**datastrdict)
        for fname in glob.glob(cdatstr + '*__vel*'):
            os.remove(fname)

    try:
        cdatstr = get_datastring(**datastrdict)

        norm_nwtnupd = dou.load_npa(cdatstr + '__norm_nwtnupd')
        vel_k = dou.load_npa(cdatstr + '__vel')
        norm_nwtnupd_list.append(norm_nwtnupd)

        print 'found vel files'
        print 'norm of last Nwtn update: {0}'.format(norm_nwtnupd)
        if norm_nwtnupd < vel_nwtn_tol:
            return vel_k, norm_nwtnupd_list

    except IOError:
        print 'no old velocity data found'
        norm_nwtnupd = 2

    if paraviewoutput:
        cdatstr = get_datastring(**datastrdict)
        vfile = dolfin.File(vfileprfx+'__steadystates.pvd')
        pfile = dolfin.File(pfileprfx+'__steadystates.pvd')
        prvoutdict = dict(V=V, Q=Q, vfile=vfile, pfile=pfile,
                          invinds=invinds, diribcs=diribcs,
                          vp=None, t=None, writeoutput=True)
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' here

    NV = A.shape[0]
    if vel_start_nwtn is None:
        vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                         rhsv=fv_stbc + fvc,
                                         rhsp=fp_stbc + fpr
                                         )

        # save the data
        cdatstr = get_datastring(**datastrdict)

        dou.save_npa(vp_stokes[:NV, ], fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_stokes))
        dou.output_paraview(**prvoutdict)

        # Stokes solution as starting value
        vel_k = vp_stokes[:NV, ]

    else:
        vel_k = vel_start_nwtn

    # Picard iterations for a good starting value for Newton
    for k in range(vel_pcrd_stps):
        (convc_mat,
         rhs_con, rhsv_conbc) = get_v_conv_conts(vel_k, invinds=invinds,
                                                 V=V, diribcs=diribcs,
                                                 Picard=True)

        vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
                                    rhsv=fv_stbc+fvc+rhs_con+rhsv_conbc,
                                    rhsp=fp_stbc + fpr)
        normpicupd = np.sqrt(m_innerproduct(M, vel_k-vp_k[:NV, :]))[0]

        print 'Picard iteration: {0} -- norm of update: {1}'\
            .format(k+1, normpicupd)

        vel_k = vp_k[:NV, ]

        if normpicupd < vel_pcrd_tol:
            break

    # Newton iteration
    while (vel_newtk < vel_nwtn_stps and norm_nwtnupd > vel_nwtn_tol):
        vel_newtk += 1

        cdatstr = get_datastring(**datastrdict)

        (convc_mat,
         rhs_con, rhsv_conbc) = get_v_conv_conts(vel_k, invinds=invinds,
                                                 V=V, diribcs=diribcs)

        vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
                                    rhsv=fv_stbc+fvc+rhs_con+rhsv_conbc,
                                    rhsp=fp_stbc + fpr)

        norm_nwtnupd = np.sqrt(m_innerproduct(M, vel_k - vp_k[:NV, :]))[0]
        vel_k = vp_k[:NV, ]
        print 'Newton iteration: {0} -- norm of update: {1}'\
            .format(vel_newtk, norm_nwtnupd)

        dou.save_npa(vel_k, fstring=cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k))
        dou.output_paraview(**prvoutdict)

    dou.save_npa(norm_nwtnupd, cdatstr + '__norm_nwtnupd')

    dou.output_paraview(**prvoutdict)

    # savetomatlab = True
    # if savetomatlab:
    #     export_mats_to_matlab(E=None, A=None, matfname='matexport')
    return vel_k, norm_nwtnupd_list


def solve_nse(A=None, M=None, J=None, JT=None,
              fvc=None, fpr=None,
              fv_stbc=None, fp_stbc=None,
              fv_tmdp=None, fv_tmdp_params={},
              fv_tmdp_memory=None,
              iniv=None, lin_vel_point=None,
              trange=None,
              t0=None, tE=None, Nts=None,
              V=None, Q=None, invinds=None, diribcs=None,
              N=None, nu=None,
              closed_loop=False, static_feedback=False,
              feedbackthroughdict=None,
              tb_mat=None, c_mat=None,
              vel_nwtn_stps=20, vel_nwtn_tol=5e-15,
              krylov=None, krpslvprms={}, krplsprms={},
              clearprvdata=False,
              get_datastring=None,
              data_prfx='',
              paraviewoutput=False, prfdir='',
              vfileprfx='', pfileprfx='',
              return_dictofvelstrs=False,
              comp_nonl_semexp=False,
              return_as_list=False,
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
    fv_tmdp : callable f(t, v, dict), optional
        time-dependent part of the right-hand side, set to zero if None
    fv_tmdp_params : dictionary, optional
        dictionary of parameters to be passed to `fv_tmdp`, defaults to `{}`
    fv_tmdp_memory : dictionary, optional
        memory of the function
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

          *preconditioner

    start_ssstokes : boolean, optional
        for your convenience, compute and use the steady state stokes solution
        as initial value, defaults to `False`


    Returns
    -------
    dictofvelstrs : dictionary, on demand
        dictionary with time `t` as keys and path to velocity files as values

    """

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if paraviewoutput:
        prvoutdict = dict(V=V, Q=Q, invinds=invinds, diribcs=diribcs,
                          vp=None, t=None, writeoutput=True)
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' here

    if trange is None:
        trange = np.linspace(t0, tE, Nts+1)

    if comp_nonl_semexp and lin_vel_point is not None:
        raise UserWarning('I am not sure what you want! ' +
                          'set either `lin_vel_point=None` ' +
                          'or `comp_nonl_semexp=False`! \n' +
                          'as it is I will compute a linear case')

    if comp_nonl_semexp:
        print 'Explicit treatment of the nonlinearity !!!'
        vel_nwtn_stps = 1

    NV = A.shape[0]

    if fv_tmdp is None:
        def fv_tmdp(time=None, curvel=None, **kw):
            return np.zeros((NV, 1)), None

    if iniv is None:
        if start_ssstokes:
            # Stokes solution as starting value
            (fv_tmdp_cont,
             fv_tmdp_memory) = fv_tmdp(time=0, **fv_tmdp_params)
            vp_stokes =\
                lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                     rhsv=fv_stbc + fvc + fv_tmdp_cont,
                                     krylov=krylov, krpslvprms=krpslvprms,
                                     krplsprms=krplsprms, rhsp=fp_stbc + fpr)
            iniv = vp_stokes[:NV]
        else:
            raise ValueError('No initial value given')

    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=trange.size-1, data_prfx=data_prfx)

    if return_as_list:
        clearprvdata = True  # we want the results at hand
    if clearprvdata:
        datastrdict['time'] = '*'
        cdatstr = get_datastring(**datastrdict)
        for fname in glob.glob(cdatstr + '__vel*'):
            os.remove(fname)

    if lin_vel_point is None:
        comp_nonl_semexp = True
        print('No linearization point given - explicit' +
              ' treatment of the nonlinearity in the first Iteration')
    else:
        cur_lin_vel_point = lin_vel_point
        # TODO: time dep linearizations

    # steady-state linearization point
    datastrdict['time'] = None
    cdatstr = get_datastring(**datastrdict)

    # TODO: this below...
    dou.save_npa(cur_lin_vel_point, fstring=cdatstr + '__vel')

    newtk, norm_nwtnupd, norm_nwtnupd_list = 0, 1, []

    # check for previously computed velocities
    try:
        datastrdict.update(dict(time=trange[-1]))
        cdatstr = get_datastring(**datastrdict)

        norm_nwtnupd = dou.load_npa(cdatstr + '__norm_nwtnupd')
        v_old = dou.load_npa(cdatstr + '__vel')

        norm_nwtnupd_list.append(norm_nwtnupd)
        print 'found vel files'
        print 'norm of last Nwtn update: {0}'.format(norm_nwtnupd)
        if norm_nwtnupd < vel_nwtn_tol and not return_dictofvelstrs:
            return
        elif norm_nwtnupd < vel_nwtn_tol:
            dictofvelstrs = {trange[0]: cdatstr + '__vel'}
            for t in trange[1:]:
                # test if the vels are there
                v_old = dou.load_npa(cdatstr + '__vel')
                # update the dict
                datastrdict.update(dict(time=t))
                cdatstr = get_datastring(**datastrdict)
                dictofvelstrs.update({t: cdatstr + '__vel'})
            return dictofvelstrs

    except IOError:
        norm_nwtnupd = 2
        print 'no old velocity data found'

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
            rhs = rhs - 0.5*dt*matvec(umat_c, matvec(vmat_c, var_c))
        else:
            umat, vmat = umat_n, vmat_n

        return solvmat, rhs, umat, vmat

    v_old = iniv  # start vector for time integration in every Newtonit
    datastrdict['time'] = trange[0]
    cdatstr = get_datastring(**datastrdict)
    if return_dictofvelstrs or not comp_nonl_semexp:
        dou.save_npa(v_old, fstring=cdatstr + '__vel')

    if return_dictofvelstrs:
        dictofvelstrs = {trange[0]: cdatstr + '__vel'}

    if return_as_list:
        vellist = []
        vellist.append(v_old)

    while (newtk < vel_nwtn_stps and norm_nwtnupd > vel_nwtn_tol):

        newtk += 1
        print 'Computing Newton Iteration {0}'.format(newtk)
        v_old = iniv  # start vector for time integration in every Newtonit

        vfile = dolfin.File(vfileprfx+'__timestep.pvd')
        pfile = dolfin.File(pfileprfx+'__timestep.pvd')
        prvoutdict.update(dict(vp=None, vc=iniv, t=trange[0],
                               pfile=pfile, vfile=vfile))
        dou.output_paraview(**prvoutdict)

        norm_nwtnupd = 0

        # ## current values_c for application of trap rule
        pcrd_anyone = newtk < 2 and vel_nwtn_stps > 1
        # use picard linearization in the first steps
        # unless solving stokes or oseen equations
        convc_mat_c, rhs_con_c, rhsv_conbc_c = \
            get_v_conv_conts(prev_v=iniv, invinds=invinds,
                             V=V, diribcs=diribcs, Picard=pcrd_anyone)
        if pcrd_anyone:
            print 'PICARD !!!'

        (fv_tmdp_cont,
         fv_tmdp_memory) = fv_tmdp(time=0,
                                   curvel=v_old,
                                   memory=fv_tmdp_memory,
                                   **fv_tmdp_params)
        fvn_c = fv_stbc + fvc + rhsv_conbc_c + rhs_con_c + fv_tmdp_cont

        if closed_loop:
            if static_feedback:
                mtxtb_c = dou.load_npa(feedbackthroughdict[None]['mtxtb'])
                next_w = dou.load_npa(feedbackthroughdict[None]['w'])
            else:
                mtxtb_c = dou.load_npa(feedbackthroughdict[0]['mtxtb'])
                next_w = dou.load_npa(feedbackthroughdict[0]['w'])

            fvn_c = fvn_c + tb_mat * (tb_mat.T * next_w)
            vmat_c = mtxtb_c.T
            try:
                umat_c = -np.array(tb_mat.todense())
            except AttributeError:
                umat_c = -tb_mat

        else:
            vmat_c = None
            umat_c = None

        print 'time to go',
        for tk, t in enumerate(trange[1:]):
            cts = t - trange[tk]
            datastrdict.update(dict(time=t))
            cdatstr = get_datastring(**datastrdict)
            sys.stdout.write("\rEnd: {1} -- now: {0:f}".format(t, trange[-1]))
            sys.stdout.flush()
            prv_datastrdict = copy.deepcopy(datastrdict)

            # coeffs and rhs at next time instance
            if pcrd_anyone or comp_nonl_semexp:
                # we rather use an explicit scheme
                # than an unstable implicit --> to get a good start value
                prev_v = v_old
            else:
                try:
                    prev_v = dou.load_npa(cdatstr + '__vel')
                except IOError:
                    prv_datastrdict['time'] = None
                    pdatstr = get_datastring(**prv_datastrdict)
                    prev_v = dou.load_npa(pdatstr + '__vel')

            if newtk == 1 and lin_vel_point is not None:
                cur_lin_vel_point = lin_vel_point
            else:
                # linearize about the prev_v value
                cur_lin_vel_point = prev_v

            convc_mat_n, rhs_con_n, rhsv_conbc_n = \
                get_v_conv_conts(prev_v=cur_lin_vel_point, invinds=invinds,
                                 V=V, diribcs=diribcs, Picard=pcrd_anyone)

            (fv_tmdp_cont,
             fv_tmdp_memory) = fv_tmdp(time=t,
                                       curvel=v_old,
                                       memory=fv_tmdp_memory,
                                       **fv_tmdp_params)
            fvn_n = fv_stbc + fvc + rhsv_conbc_n + rhs_con_n + fv_tmdp_cont

            if closed_loop:
                if static_feedback:
                    mtxtb_n = dou.load_npa(feedbackthroughdict[None]['mtxtb'])
                    next_w = dou.load_npa(feedbackthroughdict[None]['w'])
                else:
                    mtxtb_n = dou.load_npa(feedbackthroughdict[t]['mtxtb'])
                    next_w = dou.load_npa(feedbackthroughdict[t]['w'])

                fvn_n = fvn_n + tb_mat * (tb_mat.T * next_w)
                vmat_n = mtxtb_n.T
                try:
                    umat_n = -np.array(tb_mat.todense())
                except AttributeError:
                    umat_n = -tb_mat

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

            vp_new = lau.solve_sadpnt_smw(amat=solvmat,
                                          jmat=J, jmatT=JT,
                                          rhsv=rhsv,
                                          rhsp=fp_stbc + fpr,
                                          krylov=krylov, krpslvprms=krpslvprms,
                                          krplsprms=krplsprms,
                                          umat=umat, vmat=vmat)

            v_old = vp_new[:NV, ]
            (umat_c, vmat_c, fvn_c,
                convc_mat_c) = umat_n, vmat_n, fvn_n, convc_mat_n

            if return_dictofvelstrs or not comp_nonl_semexp:
                dou.save_npa(v_old, fstring=cdatstr + '__vel')
            if return_dictofvelstrs:
                dictofvelstrs.update({t: cdatstr + '__vel'})
            if return_as_list:
                vellist.append(v_old)

            prvoutdict.update(dict(vp=vp_new, t=t))
            dou.output_paraview(**prvoutdict)

            # integrate the Newton error
            norm_nwtnupd += cts * m_innerproduct(M, v_old - prev_v)

        dou.save_npa(norm_nwtnupd, cdatstr + '__norm_nwtnupd')
        norm_nwtnupd_list.append(norm_nwtnupd[0])

        print '\nnorm of current Newton update: {}'.format(norm_nwtnupd)

    if return_dictofvelstrs:
        return dictofvelstrs
    elif return_as_list:
        return vellist
    else:
        return
