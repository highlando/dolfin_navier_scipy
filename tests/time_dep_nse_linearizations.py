import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2})
krylovdict = {}

ddir = 'data/'


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0):

    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='cylinderwake', N=N, Re=Re,
                          mergerhs=True)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd)  # adding fvc, fpr
    soldict.update(tips)  # adding time integration params

    nnewtsteps = 8  # n nwtn stps for vel comp
    soldict.update(N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   data_prfx=ddir+problem,
                   paraviewoutput=False,
                   clearprvdata=True)
    snu.solve_nse(**soldict)

    nnewtsteps = 1  # n nwtn stps for vel comp
    soldict.update(N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   data_prfx=ddir+problem,
                   clearprvdata=True,
                   return_dictofvelstrs=True)
    csd = snu.solve_nse(**soldict)

    print('1, 2, check, check')

    nnewtsteps = 7  # n nwtn stps for vel comp
    soldict.update(N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   data_prfx=ddir+problem,
                   clearprvdata=False,
                   lin_vel_point=csd,
                   vel_pcrd_stps=0,
                   return_dictofvelstrs=True)

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)


if __name__ == '__main__':
    # testit(N=15, nu=1e-2)
    testit(problem='cylinderwake', N=1, Re=40, Nts=5e1, tE=0.5,
           ParaviewOutput=True)
