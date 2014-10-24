import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'

krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2,
                                              'convstatsl': [],
                                              'maxiter': 200})
# krylovdict = {}


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0):

    nnewtsteps = 9  # n nwtn stps for vel comp
    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd_vfrc, \
        rhsd_stbc, data_prfx, ddir, proutdir \
        = dnsps.get_sysmats(problem=problem, N=N, nu=nu)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(tips)  # adding time integration params
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   get_datastring=None,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   clearprvdata=True,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)
    raise Warning('TODO: debug')
    print krylovdict['krpslvprms']['convstatsl']


if __name__ == '__main__':
    # testit(N=15, nu=1e-2)
    testit(problem='cylinderwake', N=0, Re=100, Nts=2e1, tE=0.1,
           ParaviewOutput=True)
