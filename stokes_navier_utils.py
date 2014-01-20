import numpy as np
import os
import glob

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou

import sadptprj_riclyap_adi.lin_alg_utils as lau


def get_datastr_snu(nwtn=None, time=None,
                    meshp=None, nu=None, Nts=None, dt=None):

    return ('Nwtnit{0}_time{1}_nu{2}_mesh{3}_Nts{4}_dt{5}').format(
        nwtn, time, nu, meshp, Nts, dt)


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
                          picardsteps=4,
                          nnewtsteps=None, vel_nwtn_tol=None,
                          clearprvdata=True,
                          vel_start_nwtn=None,
                          ddir=None, get_datastring=None,
                          paraviewoutput=False, prfdir='', prfprfx='',
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
    :param picardsteps:
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
    :param prfprfx:
        prefix for the output files
    """

    if get_datastring is None:
        get_datastring = get_datastr_snu

    if paraviewoutput:
        curwd = os.getcwd()
        try:
            os.chdir(prfdir)
            for fname in glob.glob(prfprfx + '*'):
                os.remove(fname)
            os.chdir(curwd)
            prvoutdict = dict(V=V, Q=Q, fstring=prfdir+prfprfx,
                              invinds=invinds, diribcs=diribcs,
                              vp=None, t=None, writeoutput=True)
        except OSError:
            raise Warning('the ' + prfdir + 'subdir for storing the' +
                          ' output does not exist. Make it yourself' +
                          'or set paraviewoutput=False')
    else:
        prvoutdict = dict(writeoutput=False)  # save 'if statements' here

    norm_nwtnupd_list = []

    NV = A.shape[0]

#
# Compute or load the uncontrolled steady state Navier-Stokes solution
#

    newtk, norm_nwtnupd = 0, 1
    # a dict to be passed to the get_datastring function
    datastrdict = dict(nwtn=newtk, time=None,
                       meshp=N, nu=nu, Nts=None, dt=None)

    if clearprvdata:
        datastrdict['nwtn'] = '*'
        cdatstr = get_datastr_snu(**datastrdict)
        for fname in glob.glob(ddir + cdatstr + '*'):
            os.remove(fname)

    while newtk < nnewtsteps:
        newtk += 1
        datastrdict.update(dict(nwtn=newtk))
        # check for previously computed velocities
        try:
            cdatstr = get_datastr_snu(**datastrdict)

            norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
            vel_k = dou.load_npa(ddir + cdatstr + '__vel')

            norm_nwtnupd_list.append(norm_nwtnupd)
            print 'loaded vel files of Newton iteration {0}'.format(newtk)
            print 'norm of current Nwtn update: {0}'.format(norm_nwtnupd[0])

        except IOError:
            newtk -= 1
            break

    if newtk == 0 and vel_start_nwtn is None:
        vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
                                         rhsv=fv_stbc + fvc,
                                         rhsp=fp_stbc + fpr
                                         )

        # save the data
        cdatstr = get_datastr_snu(**datastrdict)

        dou.save_npa(vp_stokes[:NV, ], fstring=ddir + cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_stokes, fstring=prfdir+prfprfx+cdatstr))
        dou.output_paraview(**prvoutdict)

        # Stokes solution as starting value
        vel_k = vp_stokes[:NV, ]

        # picard iterations for a better newton starting value
        for k in range(picardsteps):
            (convc_mat,
             rhs_con, rhsv_conbc) = get_v_conv_conts(vel_k, invinds=invinds,
                                                     V=V, diribcs=diribcs,
                                                     Picard=True)

            vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
                                        rhsv=fv_stbc+fvc+rhs_con+rhsv_conbc,
                                        rhsp=fp_stbc + fpr)

            print 'Picard iteration: {0} -- norm of update: {1}'\
                .format(k+1, np.sqrt(m_innerproduct(M, vel_k-vp_k[:NV, :]))[0])
            vel_k = vp_k[:NV, ]

    elif newtk == 0:
        vel_k = vel_start_nwtn

    while (newtk < nnewtsteps and norm_nwtnupd > vel_nwtn_tol):
        newtk += 1

        datastrdict.update(dict(nwtn=newtk))
        cdatstr = get_datastr_snu(**datastrdict)

        print 'Computing Newton Iteration {0} -- steady state'.\
            format(newtk)

        (convc_mat,
         rhs_con, rhsv_conbc) = get_v_conv_conts(vel_k, invinds=invinds,
                                                 V=V, diribcs=diribcs)

        vp_k = lau.solve_sadpnt_smw(amat=A+convc_mat, jmat=J, jmatT=JT,
                                    rhsv=fv_stbc+fvc+rhs_con+rhsv_conbc,
                                    rhsp=fp_stbc + fpr)

        norm_nwtnupd = np.sqrt(m_innerproduct(M, vel_k - vp_k[:NV, :]))[0]
        vel_k = vp_k[:NV, ]

        dou.save_npa(vel_k, fstring=ddir + cdatstr + '__vel')

        prvoutdict.update(dict(vp=vp_k, fstring=prfdir+prfprfx+cdatstr))
        dou.output_paraview(**prvoutdict)

        dou.save_npa(norm_nwtnupd, ddir + cdatstr + '__norm_nwtnupd')
        norm_nwtnupd_list.append(norm_nwtnupd[0])

        print 'norm of current Newton update: {}'.format(norm_nwtnupd)

    savetomatlab = True
    if savetomatlab:
        export_mats_to_matlab(E=None, A=None, matfname='matexport')


    return vel_k, norm_nwtnupd_list


#def solve_nse(A=None, J=None, JT=None,
#              fvc=None, fpr=None,
#              fv_stbc=None, fp_stbc=None,
#              iniv=None, lin_vel_point=None,
#              V=None, Q=None, invinds=None, diribcs=None,
#              N=None, nu=None,
#              z_ssfeedb=None,
#              tb_mat=None, c_mat=None,
#              nnewtsteps=None, vel_nwtn_tol=None,
#              ddir=None, get_datastring=None,
#              paraviewoutput=False, prfdir='', prfprfx='',
#              **kw):
#    """
#    solution of the time-dependent nonlinear Navier-Stokes equation
#
#    using a Newton scheme in function space
#
#    """
#
#    NV = A.shape[0]
#    if iniv is None:
#        # Stokes solution as starting value
#        vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J, jmatT=JT,
#                                         rhsv=fv_stbc + fvc,
#                                         rhsp=fp_stbc + fpr)
#        iniv = vp_stokes[:NV]
#
#    if lin_vel_point is None:
#        # linearize about Stokes solution
#        lin_vel_point = vp_stokes[:NV]
#
#
#    norm_nwtnupd = 1
#    while newtk < tip['nnewtsteps']:
#        newtk += 1
#        # check for previously computed velocities
#        try:
#            cdatstr = get_datastr(nwtn=newtk, time=None,
#                                  meshp=N, timps=tip)
#
#            norm_nwtnupd = dou.load_npa(ddir + cdatstr + '__norm_nwtnupd')
#            vel_k = dou.load_npa(ddir + cdatstr + '__vel')
#
#            tip['norm_nwtnupd_list'].append(norm_nwtnupd)
#            print 'found vel files of Newton iteration {0}'.format(newtk)
#            print 'norm of current Nwtn update: {0}'.format(norm_nwtnupd[0])
#
#        except IOError:
#            newtk -= 1
#            break
#
#    while (newtk < tip['nnewtsteps'] and
#           norm_nwtnupd > tip['vel_nwtn_tol']):
#        newtk += 1
#
#        cdatstr = get_datastr(nwtn=newtk, time=None,
#                              meshp=N, timps=tip)
#
#        set_vpfiles(tip, fstring=('results/' +
#                                  'NewtonIt{0}').format(newtk))
#
#        dou.output_paraview(tip, femp, vp=vp_stokes, t=0)
#
#        norm_nwtnupd = 0
#        v_old = inivalvec  # start vector in every Newtonit
#        print 'Computing Newton Iteration {0} -- steady state'.\
#            format(newtk)
#
#        for t in np.linspace(tip['t0']+DT, tip['tE'], Nts):
#            cdatstr = get_datastr(nwtn=newtk, time=t,
#                                  meshp=N, timps=tip)
#
#            # t for implicit scheme
#            pdatstr = get_datastr(nwtn=newtk-1, time=t,
#                                  meshp=N, timps=tip)
#
#            # try - except for linearizations about stationary sols
#            # for which t=None
#            try:
#                prev_v = dou.load_npa(ddir + pdatstr + '__vel')
#            except IOError:
#                pdatstr = get_datastr(nwtn=newtk - 1, time=None,
#                                      meshp=N, timps=tip)
#                prev_v = dou.load_npa(ddir + pdatstr + '__vel')
#
#            convc_mat, rhs_con, rhsv_conbc = get_v_conv_conts(prev_v,
#                                                              femp, tip)
#
#            rhsd_cur = dict(fv=stokesmatsc['M'] * v_old +
#                            DT * (rhs_con[INVINDS, :] +
#                                  rhsv_conbc + rhsd_vfstbc['fv']),
#                            fp=rhsd_vfstbc['fp'])
#
#            matd_cur = dict(A=stokesmatsc['M'] +
#                            DT * (stokesmatsc['A'] + convc_mat),
#                            JT=stokesmatsc['JT'],
#                            J=stokesmatsc['J'])
#
#            vp = lau.stokes_steadystate(matdict=matd_cur,
#                                        rhsdict=rhsd_cur)
#
#            v_old = vp[:NV, ]
#
#            dou.save_npa(v_old, fstring=ddir + cdatstr + '__vel')
#
#            dou.output_paraview(tip, femp, vp=vp, t=t),
#
#            # integrate the Newton error
#            norm_nwtnupd += DT * np.dot((v_old - prev_v).T,
#                                        stokesmatsc['M'] *
#                                        (v_old - prev_v))
#
#        dou.save_npa(norm_nwtnupd, ddir + cdatstr + '__norm_nwtnupd')
#        tip['norm_nwtnupd_list'].append(norm_nwtnupd[0])
#
#        print 'norm of current Newton update: {}'.format(norm_nwtnupd)
