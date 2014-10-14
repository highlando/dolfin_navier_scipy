import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2})
krylovdict = {}


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0):

    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd_vfrc, \
        rhsd_stbc, data_prfx, ddir, proutdir \
        = dnsps.get_sysmats(problem=problem, N=N, nu=nu)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(tips)  # adding time integration params

    nnewtsteps = 8  # n nwtn stps for vel comp
    soldict.update(fv_stbc=rhsd_stbc['fv'], fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=False,
                   clearprvdata=True)
    snu.solve_nse(**soldict)

    nnewtsteps = 1  # n nwtn stps for vel comp
    soldict.update(fv_stbc=rhsd_stbc['fv'],
                   fp_stbc=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   data_prfx=ddir+data_prfx,
                   clearprvdata=True,
                   return_dictofvelstrs=True)
    csd = snu.solve_nse(**soldict)

    print '1, 2, check, check'

    nnewtsteps = 7  # n nwtn stps for vel comp
    csd = soldict.update(fv_stbc=rhsd_stbc['fv'],
                         fp_stbc=rhsd_stbc['fp'],
                         N=N, nu=nu,
                         vel_nwtn_stps=nnewtsteps,
                         vel_nwtn_tol=vel_nwtn_tol,
                         start_ssstokes=True,
                         data_prfx=ddir+data_prfx,
                         clearprvdata=False,
                         lin_vel_point=csd,
                         vel_pcrd_stps=0,
                         return_dictofvelstrs=True)

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)


if __name__ == '__main__':
    # testit(N=15, nu=1e-2)
    testit(problem='cylinderwake', N=0, Re=100, Nts=5e1, tE=0.5,
           ParaviewOutput=True)
