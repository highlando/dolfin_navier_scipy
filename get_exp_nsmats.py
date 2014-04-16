import dolfin
import os
import numpy as np
import datetime
import scipy.io


import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.data_output_utils as dou
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import sadptprj_riclyap_adi.lin_alg_utils as lau

import distr_control_fenics.cont_obs_utils as cou

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def comp_exp_nsmats(problemname='drivencavity',
                    N=10, Re=1e2, nu=None,
                    linear_system=False,
                    mddir='pathtodatastorage'):

    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc, \
        data_prfx, ddir, proutdir = \
        dnsps.get_sysmats(problem=problemname, N=N, Re=Re)

    invinds = femp['invinds']
    A, J, M = stokesmatsc['A'], stokesmatsc['J'], stokesmatsc['M']
    fvc, fpc = rhsd_vfrc['fvc'], rhsd_vfrc['fpr']
    fv_stbc, fp_stbc = rhsd_stbc['fv'], rhsd_stbc['fp']
    invinds = femp['invinds']
    NV, NP = invinds.shape[0], J.shape[0]

    data_prfx = problemname + '__'
    NU, NY = 3, 4

    # specify in what spatial direction Bu changes. The remaining is constant
    if problemname == 'drivencavity':
        uspacedep = 0
    elif problemname == 'cylinderwake':
        uspacedep = 1

    # output
    ddir = 'data/'
    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

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

    cdatstr = snu.get_datastr_snu(time=None, meshp=N, nu=nu, Nts=None)

    (coors, xinds,
     yinds, corfunvec) = dts.get_dof_coors(femp['V'], invinds=invinds)

    ctrl_visu_str = \
        ' the control setup is as follows \n' +\
        ' B maps into the domain of control -' +\
        ' the first half of the columns' +\
        'actuate in x-direction, the second in y direction \n' +\
        ' C measures averaged velocities in the domain of observation' +\
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

    if linear_system:
        soldict = stokesmatsc  # containing A, J, JT
        soldict.update(femp)  # adding V, Q, invinds, diribcs
        soldict.update(rhsd_vfrc)  # adding fvc, fpr
        soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                       N=N, nu=nu,
                       get_datastring=None,
                       data_prfx=ddir+data_prfx,
                       paraviewoutput=False
                       )

        # compute the uncontrolled steady state Stokes solution
        v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)

        convc_mat, rhs_con, rhsv_conbc = \
            snu.get_v_conv_conts(prev_v=v_ss_nse, invinds=invinds,
                                 V=femp['V'], diribcs=femp['diribcs'])
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

        scipy.io.savemat(mddir + problemname +
                         '__mats_N{0}_Re{1}'.format(NV, Re),
                         dict(A=f_mat, M=stokesmatsc['M'],
                              nu=femp['nu'], Re=femp['Re'],
                              J=stokesmatsc['J'], B=b_mat, C=c_mat,
                              v_ss_nse=v_ss_nse, info=infostr,
                              contsetupstr=contsetupstr, datastr=cdatstr,
                              coors=coors, xinds=xinds, yinds=yinds,
                              corfunvec=corfunvec))

    else:
        print 'assembling hmat ...'
        hmat = dts.ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)

        zerv = np.zeros((NV, 1))
        bc_conv, bc_rhs_conv, rhsbc_convbc = \
            snu.get_v_conv_conts(prev_v=zerv, V=femp['V'], invinds=invinds,
                                 diribcs=femp['diribcs'], Picard=False)

        f_mat = - stokesmatsc['A'] - bc_conv
        fv = fv_stbc + fvc + bc_rhs_conv + rhsbc_convbc
        fp = fp_stbc + fpc

        infostr = 'These are the coefficient matrices of the quadratic ' +\
            'formulation of the Navier-Stokes Equations \n for the ' +\
            problemname + ' to be used as \n\n' +\
            ' $M \\dot v = Av + H*kron(v,v) + J^Tp + Bu + fv$ \n' +\
            ' and  $Jv = fp$ \n\n' +\
            ' the Reynoldsnumber is computed as L/nu \n' +\
            ' note that `A` contains the diffusion and the linear term \n' +\
            ' that comes from the dirichlet boundary values \n' +\
            ' see https://github.com/highlando/dolfin_navier_scipy/blob/' +\
            ' master/tests/solve_nse_quadraticterm.py for appl example\n' +\
            ctrl_visu_str

    scipy.io.savemat(mddir + problemname +
                     'quadform__mats_N{0}_Re{1}'.format(NV, Re),
                     dict(A=f_mat, M=stokesmatsc['M'],
                          H=hmat, fv=fv, fp=fp,
                          nu=femp['nu'], Re=femp['Re'],
                          J=stokesmatsc['J'], B=b_mat, C=c_mat,
                          v_ss_nse=v_ss_nse, info=infostr,
                          contsetupstr=contsetupstr, datastr=cdatstr,
                          coors=coors, xinds=xinds, yinds=yinds,
                          corfunvec=corfunvec))


if __name__ == '__main__':
    mddir = '/afs/mpi-magdeburg.mpg.de/data/csc/projects/qbdae-nse/data/'
    comp_exp_nsmats(problemname='drivencavity', N=10, Re=1e-2,
                    savetomatfiles=False, mddir=mddir, linear_system=False)
