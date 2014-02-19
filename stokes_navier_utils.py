import numpy as np
import os
import glob
import copy
import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def get_datastr_snu(time=None, meshp=None, nu=None, Nts=None, dt=None,
                    data_prfx=''):

    return (data_prfx +
            'time{0}_nu{1}_mesh{2}_Nts{3}_dt{4}').format(
        time, nu, meshp, Nts, dt)


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

    :return:
    ``N1`` matrix representing :math:`(u_0 \\cdot \\nabla )u`
    ``N2`` matrix representing :math:`(u \\cdot \\nabla )u_0`
    ``fv`` vector representing :math:`(u_0 \\cdot \\nabla )u_0`

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
    """ inner product with a spd matrix

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
                          vel_nwtn_stps=20, vel_nwtn_tol=1e-15,
                          clearprvdata=False,
                          vel_start_nwtn=None,
                          ddir=None, get_datastring=None,
                          data_prfx='',
                          paraviewoutput=False,
                          save_intermediate_steps=False,
                          vfileprfx='', pfileprfx='',
                          **kw):

    """
    Solution of the steady state nonlinear NSE Problem

    using Newton's scheme. If no starting value is provide, the iteration
    is started with the steady state Stokes solution.

    :param fvc, fpr:
        right hand sides restricted via removing the boundary nodes in the
        momentum and the pressure freedom in the continuity equation
    :param fv_stbc, fp_stbc:
        contributions to the right hand side by the Dirichlet boundary
        conditions in the stokes equations. TODO: time dependent conditions
        are not handled by now
    :param npicardsteps:
        Number of Picard iterations when computing a starting value for the
        Newton scheme, cf. Elman, Silvester, Wathen: *FEM and fast iterative
        solvers*, 2005
    :param ddir:
        path to directory where the data is stored
    :param get_datastring:
        routine that returns a string describing the data
    :param paraviewoutput:
        boolean control whether paraview output is produced
    :param prfdir:
        path to directory where the paraview output is stored
    :param pfileprfx, vfileprfx:
        prefix for the output files
    """

    if get_datastring is None:
        get_datastring = get_datastr_snu

#
# Compute or load the uncontrolled steady state Navier-Stokes solution
#

    norm_nwtnupd_list = []
    vel_newtk, norm_nwtnupd = 0, 1
    # a dict to be passed to the get_datastring function
    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=None, dt=None, data_prfx=data_prfx)

    if clearprvdata:
        cdatstr = get_datastr_snu(**datastrdict)
        for fname in glob.glob(ddir + cdatstr + '*__vel*'):
            os.remove(fname)

    try:
        cdatstr = get_datastr_snu(**datastrdict)

        norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
        vel_k = dou.load_npa(ddir + cdatstr + '__vel')

        norm_nwtnupd_list.append(norm_nwtnupd)
        print 'found vel files'
        print 'norm of last Nwtn update: {0}'.format(norm_nwtnupd[0])
        if norm_nwtnupd < vel_nwtn_tol:
            return vel_k, norm_nwtnupd_list

    except IOError:
        print 'no old velocity data found'

    if paraviewoutput:
        cdatstr = get_datastr_snu(**datastrdict)
        vfile = dolfin.File(vfileprfx+cdatstr+'__steadystates.pvd')
        pfile = dolfin.File(pfileprfx+cdatstr+'__steadystates.pvd')
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
        cdatstr = get_datastr_snu(**datastrdict)

        dou.save_npa(vp_stokes[:NV, ], fstring=ddir + cdatstr + '__vel')

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

        cdatstr = get_datastr_snu(**datastrdict)

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

        dou.save_npa(vel_k, fstring=ddir + cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k))
        dou.output_paraview(**prvoutdict)

    # savetomatlab = True
    # if savetomatlab:
    #     export_mats_to_matlab(E=None, A=None, matfname='matexport')
    return vel_k, norm_nwtnupd_list


def solve_nse(A=None, M=None, J=None, JT=None,
              fvc=None, fpr=None,
              fv_stbc=None, fp_stbc=None,
              iniv=None, lin_vel_point=None,
              trange=None,
              t0=None, tE=None, Nts=None,
              V=None, Q=None, invinds=None, diribcs=None,
              N=None, nu=None,
              closed_loop=False, static_feedback=False,
              feedbackthroughdict=None,
              tb_mat=None, c_mat=None,
              vel_nwtn_stps=20, vel_nwtn_tol=1e-15,
              clearprvdata=False,
              ddir=None, get_datastring=None,
              data_prfx='',
              paraviewoutput=False, prfdir='',
              vfileprfx='', pfileprfx='',
              return_nwtn_step=False,
              **kw):
    """
    solution of the time-dependent nonlinear Navier-Stokes equation

    using a Newton scheme in function space



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

    NV = A.shape[0]

    if iniv is None:
        # Stokes solution as starting value
        vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                         rhsv=fv_stbc + fvc,
                                         rhsp=fp_stbc + fpr)
        iniv = vp_stokes[:NV]

    datastrdict = dict(time=None, meshp=N, nu=nu,
                       Nts=trange.size-1, dt=None, data_prfx=data_prfx)

    if clearprvdata:
        datastrdict['time'] = '*'
        cdatstr = get_datastr_snu(**datastrdict)
        for fname in glob.glob(ddir + cdatstr + '__vel*'):
            os.remove(fname)

    if lin_vel_point is None:
        # linearize about the initial value
        datastrdict['time'] = None
        cdatstr = get_datastr_snu(**datastrdict)
        lin_vel_point = iniv
        dou.save_npa(iniv, fstring=ddir + cdatstr + '__vel')

    newtk, norm_nwtnupd, norm_nwtnupd_list = 0, 1, []

    # check for previously computed velocities
    try:
        datastrdict.update(dict(time=trange[-1], dt=trange[-1]-trange[-2]))
        cdatstr = get_datastr_snu(**datastrdict)

        norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
        v_old = dou.load_npa(ddir + cdatstr + '__vel')

        norm_nwtnupd_list.append(norm_nwtnupd)
        print 'found vel files'
        print 'norm of last Nwtn update: {0}'.format(norm_nwtnupd[0])
        if norm_nwtnupd < vel_nwtn_tol:
            return

    except IOError:
        print 'no old velocity data found'

    while (newtk < vel_nwtn_stps and norm_nwtnupd > vel_nwtn_tol):
        newtk += 1
        vfile = dolfin.File(vfileprfx+cdatstr+'__timestep.pvd')
        pfile = dolfin.File(pfileprfx+cdatstr+'__timestep.pvd')
        prvoutdict.update(dict(vp=None, vc=iniv, t=trange[0],
                               pfile=pfile, vfile=vfile))
        dou.output_paraview(**prvoutdict)

        norm_nwtnupd = 0
        v_old = iniv  # start vector for time integration in every Newtonit
        print 'Computing Newton Iteration {0}'.format(newtk)

        for tk, t in enumerate(trange[1:]):
            cts = t - trange[tk]
            datastrdict.update(dict(time=t, dt=cts))
            cdatstr = get_datastr_snu(**datastrdict)

            prv_datastrdict = copy.deepcopy(datastrdict)
            # t for implicit scheme
            # prv_datastrdict['nwtn'], prv_datastrdict['time'] = newtk-1, t
            # pdatstr = get_datastr_snu(**prv_datastrdict)

            # try - except for linearizations about stationary sols
            # for which t=None
            try:
                prev_v = dou.load_npa(ddir + cdatstr + '__vel')
            except IOError:
                prv_datastrdict['time'], prv_datastrdict['dt'] = None, None
                pdatstr = get_datastr_snu(**prv_datastrdict)
                prev_v = dou.load_npa(ddir + pdatstr + '__vel')

            convc_mat, rhs_con, rhsv_conbc = \
                get_v_conv_conts(prev_v=prev_v, invinds=invinds,
                                 V=V, diribcs=diribcs)

            fvn = fv_stbc + fvc + rhsv_conbc + rhs_con

            if closed_loop:
                if static_feedback:
                    mtxtb = dou.load_npa(feedbackthroughdict[None]['mtxtb'])
                    next_w = dou.load_npa(feedbackthroughdict[None]['w'])
                else:
                    mtxtb = dou.load_npa(feedbackthroughdict[t]['mtxtb'])
                    next_w = dou.load_npa(feedbackthroughdict[t]['w'])

                fvn = fvn + tb_mat * (tb_mat.T * next_w)
                vmat = mtxtb
                umat = -cts*tb_mat

            else:
                vmat = None
                umat = None

            vp_new = lau.solve_sadpnt_smw(amat=M + cts*(A + convc_mat),
                                          jmat=J, jmatT=JT,
                                          rhsv=M*v_old + cts * fvn,
                                          rhsp=fp_stbc + fpr,
                                          umat=umat, vmat=vmat)

            v_old = vp_new[:NV, ]

            dou.save_npa(v_old, fstring=ddir + cdatstr + '__vel')

            prvoutdict.update(dict(vp=vp_new, t=t))
                                   # fstring=prfdir+data_prfx+cdatstr))
            dou.output_paraview(**prvoutdict)

            # integrate the Newton error
            norm_nwtnupd += cts * m_innerproduct(M, v_old - prev_v)

        dou.save_npa(norm_nwtnupd, ddir + cdatstr + '__norm_nwtnupd')
        norm_nwtnupd_list.append(norm_nwtnupd[0])

        print 'norm of current Newton update: {}'.format(norm_nwtnupd)
