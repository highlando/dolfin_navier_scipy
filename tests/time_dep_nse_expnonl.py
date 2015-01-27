import dolfin
import os

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2})
krylovdict = {}


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None,
           t0=0.0, tE=1.0, Nts=1e2+1, scheme=None):

    problemdict = dict(drivencavity=dnsps.drivcav_fems,
                       cylinderwake=dnsps.cyl_fems)
    problemfem = problemdict[problem]
    femp = problemfem(N, scheme=scheme)

    dolfin.plot(femp['V'].mesh())

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re

    nnewtsteps = 9  # n nwtn stps for vel comp
    vel_nwtn_tol = 1e-14
    # prefix for data files
    data_prfx = problem
    # dir to store data
    ddir = 'data/'
    # paraview output
    ParaviewOutput = True
    proutdir = 'results/'
    tips = dict(t0=t0, tE=tE, Nts=Nts)

    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    if ParaviewOutput:
        curwd = os.getcwd()
        try:
            os.chdir(proutdir)
            os.chdir(curwd)
        except OSError:
            raise Warning('the ' + proutdir + ' subdir for storing the' +
                          ' output does not exist. Make it yourself' +
                          ' or set paraviewoutput=False')

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'], nu)

    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure
    # if required
    if problem == 'cylinderwake':
        ppin = None
    else:
        ppin = -1
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
        rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]

    # reduce the matrices by resolving the BCs
    (stokesmatsc,
     rhsd_stbc,
     invinds,
     bcinds,
     bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                         femp['diribcs'])

    print stokesmatsc['J'].shape
    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds,
              'bcvals': bcvals,
              'invinds': invinds}
    femp.update(bcdata)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd_stbc['fv'], fp=rhsd_stbc['fp'],
                   N=N, nu=nu, ppin=ppin,
                   start_ssstokes=True,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   get_datastring=None,
                   comp_nonl_semexp=True,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

    soldict.update(krylovdict)  # if we wanna use an iterative solver

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    # v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)
    snu.solve_nse(**soldict)


if __name__ == '__main__':
    # testit(N=15, nu=1e-3)
    testit(problem='cylinderwake', N=3, Re=60, t0=0.0, tE=2., Nts=512,
           scheme='CR')
