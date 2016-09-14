# import dolfin
import numpy as np
import datetime
import scipy.io


import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau

import distr_control_fenics.cont_obs_utils as cou

# dolfin.parameters.linear_algebra_backend = 'uBLAS'

NU, NY = 7, 3
ddir = 'data/'


def comp_exp_nsmats(problemname='drivencavity',
                    N=10, Re=1e2, nu=None,
                    linear_system=False, refree=False,
                    bccontrol=False, palpha=None,
                    use_old_data=False,
                    mddir='pathtodatastorage'):
    """compute and export the system matrices for Navier-Stokes equations

    Parameters
    ---
    refree : boolean, optional
        whether to use `Re=1` (so that the `Re` number can be applied later by
        scaling the corresponding matrices, defaults to `False`
    linear_system : boolean, optional
        whether to compute/return the linearized system, defaults to `False`
    bccontrol : boolean, optional
        whether to model boundary control at the cylinder via penalized robin
        boundary conditions, defaults to `False`
    palpha : float, optional
        penalization parameter for the boundary control, defaults to `None`,
        `palpha` is mandatory for `linear_system`

    """

    if refree:
        Re = 1
        print 'For the Reynoldsnumber free mats, we set Re=1'

    if problemname == 'drivencavity' and bccontrol:
        raise NotImplementedError('boundary control for the driven cavity' +
                                  ' is not implemented yet')

    if linear_system and bccontrol and palpha is None:
        raise UserWarning('For the linear system a' +
                          ' value for `palpha` is needed')
    if not linear_system and bccontrol:
        raise NotImplementedError('Nonlinear system with boundary control' +
                                  ' is not implemented yet')
    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc = \
        dnsps.get_sysmats(problem=problemname, bccontrol=bccontrol, N=N, Re=Re)
    if linear_system and bccontrol:
        Arob = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
        Brob = 1./palpha*stokesmatsc['Brob']
    elif linear_system:
        Brob = 0

    invinds = femp['invinds']
    A, J = stokesmatsc['A'], stokesmatsc['J']
    fvc, fpc = rhsd_vfrc['fvc'], rhsd_vfrc['fpr']
    fv_stbc, fp_stbc = rhsd_stbc['fv'], rhsd_stbc['fp']
    invinds = femp['invinds']
    NV = invinds.shape[0]
    data_prfx = problemname + '__N{0}Re{1}'.format(N, Re)
    if bccontrol:
        data_prfx = data_prfx + '_penarob'

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    fv = rhsd_vfrc['fvc'] + rhsd_stbc['fv']
    fp = rhsd_vfrc['fpr'] + rhsd_stbc['fp']
    # print 'get expmats: ||fv|| = {0}'.format(np.linalg.norm(fv))
    # print 'get expmats: ||fp|| = {0}'.format(np.linalg.norm(fp))
    # import scipy.sparse.linalg as spsla
    # print 'get expmats: ||A|| = {0}'.format(spsla.norm(A))
    # print 'get expmats: ||Arob|| = {0}'.format(spsla.norm(Arob))
    # print 'get expmats: ||A|| = {0}'.format(spsla.norm(stokesmatsc['A']))
    # raise Warning('TODO: debug')

    soldict.update(fv=fv, fp=fp,
                   N=N, nu=nu,
                   clearprvdata=~use_old_data,
                   get_datastring=None,
                   data_prfx=ddir+data_prfx+'_stst',
                   paraviewoutput=False
                   )
    if bccontrol and linear_system:
        soldict.update(A=Arob)

    # compute the uncontrolled steady state Navier-Stokes solution
    vp_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(return_vp=True,
                                                             **soldict)
    v_ss_nse, p_ss_nse = vp_ss_nse[:NV], vp_ss_nse[NV:]

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        uspacedep = 0
    elif problemname == 'cylinderwake':
        uspacedep = 1

    #
    # Control mats
    #
    contsetupstr = problemname + '__NV{0}NU{1}NY{2}'.format(NV, NU, NY)

    # get the control and observation operators
    try:
        b_mat = dou.load_spa(ddir + contsetupstr + '__b_mat')
        u_masmat = dou.load_spa(ddir + contsetupstr + '__u_masmat')
        print 'loaded `b_mat`'
    except IOError:
        print 'computing `b_mat`...'
        b_mat, u_masmat = cou.get_inp_opa(cdcoo=femp['cdcoo'], V=femp['V'],
                                          NU=NU, xcomp=uspacedep)
        dou.save_spa(b_mat, ddir + contsetupstr + '__b_mat')
        dou.save_spa(u_masmat, ddir + contsetupstr + '__u_masmat')
    try:
        mc_mat = dou.load_spa(ddir + contsetupstr + '__mc_mat')
        y_masmat = dou.load_spa(ddir + contsetupstr + '__y_masmat')
        print 'loaded `c_mat`'
    except IOError:
        print 'computing `c_mat`...'
        mc_mat, y_masmat = cou.get_mout_opa(odcoo=femp['odcoo'],
                                            V=femp['V'], NY=NY)
        dou.save_spa(mc_mat, ddir + contsetupstr + '__mc_mat')
        dou.save_spa(y_masmat, ddir + contsetupstr + '__y_masmat')

    # restrict the operators to the inner nodes
    mc_mat = mc_mat[:, invinds][:, :]
    b_mat = b_mat[invinds, :][:, :]

    c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
    # TODO: right choice of norms for y
    #       and necessity of regularization here
    #       by now, we go on number save

    # the pressure observation mean over a small domain
    if problemname == 'cylinderwake':
        podcoo = dict(xmin=0.6,
                      xmax=0.64,
                      ymin=0.18,
                      ymax=0.22)
    elif problemname == 'drivencavity':
        podcoo = dict(xmin=0.45,
                      xmax=0.55,
                      ymin=0.7,
                      ymax=0.8)
    else:
        podcoo = femp['odcoo']

    # description of the control and observation domains
    dmd = femp['cdcoo']
    xmin, xmax, ymin, ymax = dmd['xmin'], dmd['xmax'], dmd['ymin'], dmd['ymax']
    velcondomstr = 'vel control domain: [{0}, {1}]x[{2}, {3}]\n'.\
        format(xmin, xmax, ymin, ymax)
    dmd = femp['odcoo']
    xmin, xmax, ymin, ymax = dmd['xmin'], dmd['xmax'], dmd['ymin'], dmd['ymax']
    velobsdomstr = 'vel observation domain: [{0}, {1}]x[{2}, {3}]\n'.\
        format(xmin, xmax, ymin, ymax)
    dmd = podcoo
    xmin, xmax, ymin, ymax = dmd['xmin'], dmd['xmax'], dmd['ymin'], dmd['ymax']
    pobsdomstr = 'pressure observation domain: [{0}, {1}]x[{2}, {3}]\n'.\
        format(xmin, xmax, ymin, ymax)

    pcmat = cou.get_pavrg_onsubd(odcoo=podcoo, Q=femp['Q'], ppin=None)

    cdatstr = snu.get_datastr_snu(time=None, meshp=N, nu=nu, Nts=None)

    (coors, xinds,
     yinds, corfunvec) = dts.get_dof_coors(femp['V'], invinds=invinds)

    ctrl_visu_str = \
        ' the (distributed) control setup is as follows \n' +\
        ' B maps into the domain of control -' +\
        velcondomstr +\
        ' the first half of the columns' +\
        'actuate in x-direction, the second in y direction \n' +\
        ' Cv measures averaged velocities in the domain of observation' +\
        velobsdomstr +\
        ' Cp measures the averaged pressure' +\
        ' in the domain of pressure observation: ' +\
        pobsdomstr +\
        ' the first components are in x, the last in y-direction \n\n' +\
        ' Visualization: \n\n' +\
        ' `coors`   -- array of (x,y) coordinates in ' +\
        ' the same order as v[xinds] or v[yinds] \n' +\
        ' `xinds`, `yinds` -- indices of x and y components' +\
        ' of v = [vx, vy] -- note that indexing starts with 0\n' +\
        ' for testing use corfunvec wich is the interpolant of\n' +\
        ' f(x,y) = [x, y] on the grid \n\n' +\
        'Created in `get_exp_nsmats.py` ' +\
        '(see https://github.com/highlando/dolfin_navier_scipy) at\n' +\
        datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")

    if bccontrol and problemname == 'cylinderwake' and linear_system:
        ctrl_visu_str = \
            ('the boundary control is realized via penalized robin \n' +
             'boundary conditions, cf. e.g. [Hou/Ravindran `98], \n' +
             'with predefined shape functions for the cylinder wake \n' +
             'and the penalization parameter `palpha`={0}.').format(palpha) +\
            ctrl_visu_str

    if linear_system:
        convc_mat, rhs_con, rhsv_conbc = \
            snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                 V=femp['V'], diribcs=femp['diribcs'])
        # TODO: omg
        if bccontrol:
            f_mat = - Arob - convc_mat
        else:
            f_mat = - stokesmatsc['A'] - convc_mat

        infostr = 'These are the coefficient matrices of the linearized ' +\
            'Navier-Stokes Equations \n for the ' +\
            problemname + ' to be used as \n\n' +\
            ' $M \\dot v = Av + J^Tp + Bu$   and  $Jv = 0$ \n\n' +\
            ' the Reynoldsnumber is computed as L/nu \n' +\
            ' Note this is the reduced system for the velocity update\n' +\
            ' caused by the control, i.e., no boundary conditions\n' +\
            ' or inhomogeneities here. To get the actual flow, superpose \n' +\
            ' the steadystate velocity solution `v_ss_nse` \n\n' +\
            ctrl_visu_str

        matstr = (mddir + problemname + '__mats_N{0}_Re{1}').format(NV, Re)
        if bccontrol:
            matstr = matstr + '__penarob_palpha{0}'.format(palpha)

        scipy.io.savemat(matstr,
                         dict(A=f_mat, M=stokesmatsc['M'],
                              nu=femp['nu'], Re=femp['Re'],
                              J=stokesmatsc['J'], B=b_mat, C=c_mat,
                              Cp=pcmat, Brob=Brob,
                              v_ss_nse=v_ss_nse, info=infostr,
                              contsetupstr=contsetupstr, datastr=cdatstr,
                              coors=coors, xinds=xinds, yinds=yinds,
                              corfunvec=corfunvec))

        print('matrices saved to ' + matstr)

    elif refree:
        hstr = ddir + problemname + '_N{0}_hmat'.format(N)
        try:
            hmat = dou.load_spa(hstr)
            print 'loaded `hmat`'
        except IOError:
            print 'assembling hmat ...'
            hmat = dts.ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)
            dou.save_spa(hmat, hstr)

        zerv = np.zeros((NV, 1))
        bc_conv, bc_rhs_conv, rhsbc_convbc = \
            snu.get_v_conv_conts(prev_v=zerv, V=femp['V'], invinds=invinds,
                                 diribcs=femp['diribcs'], Picard=False)

        # diff_mat = stokesmatsc['A']
        # bcconv_mat = bc_conv
        # fv_bcdiff = fv_stbc
        # fv_bcconv = - bc_rhs_conv
        fv = fvc
        fp = fpc
        # fp_bc = fp_stbc

        infostr = 'These are the coefficient matrices of the quadratic ' +\
            'formulation of the Navier-Stokes Equations \n for the ' +\
            problemname + ' to be used as \n\n' +\
            ' $M \\dot v + Av + H*kron(v,v) + J^Tp = Bu + fv$ \n' +\
            ' and  $Jv = fp$ \n\n' +\
            ' the Reynoldsnumber is computed as L/nu \n' +\
            ' note that `A` contains the diffusion and the linear term \n' +\
            ' that comes from the dirichlet boundary values \n' +\
            ' as initial value one can use the provided steady state \n' +\
            ' Stokes solution \n' +\
            ' see https://github.com/highlando/dolfin_navier_scipy/blob/' +\
            ' master/tests/solve_nse_quadraticterm.py for appl example\n' +\
            ctrl_visu_str

        scipy.io.savemat(mddir + problemname +
                         'quadform__mats_N{0}_Re{1}'.format(NV, Re),
                         dict(A=f_mat, M=stokesmatsc['M'],
                              H=-hmat, fv=fv, fp=fp,
                              nu=femp['nu'], Re=femp['Re'],
                              J=stokesmatsc['J'], B=b_mat, Cv=c_mat,
                              Cp=pcmat,
                              info=infostr,
                              # ss_stokes=old_v,
                              contsetupstr=contsetupstr, datastr=cdatstr,
                              coors=coors, xinds=xinds, yinds=yinds,
                              corfunvec=corfunvec))
    else:
        hstr = ddir + problemname + '_N{0}_hmat'.format(N)
        try:
            hmat = dou.load_spa(hstr)
            print 'loaded `hmat`'
        except IOError:
            print 'assembling hmat ...'
            hmat = dts.ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)
            dou.save_spa(hmat, hstr)

        zerv = np.zeros((NV, 1))
        bc_conv, bc_rhs_conv, rhsbc_convbc = \
            snu.get_v_conv_conts(prev_v=zerv, V=femp['V'], invinds=invinds,
                                 diribcs=femp['diribcs'], Picard=False)

        f_mat = - stokesmatsc['A'] - bc_conv
        l_mat = -bc_conv
        fv = fv_stbc + fvc - bc_rhs_conv
        fp = fp_stbc + fpc

        vp_stokes = lau.solve_sadpnt_smw(amat=A, jmat=J,
                                         rhsv=fv_stbc + fvc,
                                         rhsp=fp_stbc + fpc)
        old_v = vp_stokes[:NV]
        p_stokes = -vp_stokes[NV:]  # the pressure was flipped for symmetry

        infostr = 'These are the coefficient matrices of the quadratic ' +\
            'formulation of the Navier-Stokes Equations \n for the ' +\
            problemname + ' to be used as \n\n' +\
            ' $M \\dot v = Av + H*kron(v,v) + J^Tp + Bu + fv$ \n' +\
            ' and  $Jv = fp$ \n\n' +\
            ' the Reynoldsnumber is computed as L/nu \n' +\
            ' note that `A` contains the diffusion and the linear term `L`\n' +\
            ' that comes from the dirichlet boundary values \n' +\
            ' for linearizations it might be necessary to consider `A-L` \n' +\
            ' as initial value one can use the provided steady state \n' +\
            ' Stokes solution \n' +\
            ' see https://github.com/highlando/dolfin_navier_scipy/blob/' +\
            ' master/tests/solve_nse_quadraticterm.py for appl example\n' +\
            ctrl_visu_str

        scipy.io.savemat(mddir + problemname +
                         'quadform__mats_N{0}_Re{1}'.format(NV, Re),
                         dict(A=f_mat, M=stokesmatsc['M'],
                              H=-hmat, fv=fv, fp=fp, L=l_mat,
                              nu=femp['nu'], Re=femp['Re'],
                              J=stokesmatsc['J'], B=b_mat, Cv=c_mat,
                              Cp=pcmat,
                              info=infostr,
                              p_ss_stokes=p_stokes, p_ss_nse=p_ss_nse,
                              v_ss_stokes=old_v, v_ss_nse=v_ss_nse,
                              contsetupstr=contsetupstr, datastr=cdatstr,
                              coors=coors, xinds=xinds, yinds=yinds,
                              corfunvec=corfunvec))


if __name__ == '__main__':
    mddir = 'data/'
    # 'afs/mpi-magdeburg.mpg.de/data/csc/projects/qbdae-nse/data/'
    comp_exp_nsmats(problemname='drivencavity', N=10, Re=1e-2,
                    mddir=mddir, linear_system=False)
