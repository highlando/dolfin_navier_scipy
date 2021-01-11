import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

# dolfin.parameters.linear_algebra_backend = 'uBLAS'
# dolfin.parameters.linear_algebra_backend = 'Eigen'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2})
krylovdict = {}


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, nonltrt=None,
           t0=0.0, tE=1.0, Nts=1e2+1, ParaviewOutput=False, scheme='TH'):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem=problem, Re=Re, nu=nu, scheme=scheme,
                          meshparams=dict(refinement_level=N), mergerhs=True)
    proutdir = 'results/'

    dolfin.plot(femp['V'].mesh())

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=N, nu=nu,
                   start_ssstokes=True,
                   treat_nonl_explicit=nonltrt,
                   no_data_caching=True,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_expnl_',
                   pfileprfx=proutdir+'p_expnl')

    soldict.update(krylovdict)  # if we wanna use an iterative solver

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    # vp_ss_nse = snu.solve_steadystate_nse(**soldict)
    # soldict.update(dict(start_ssstokes=True))
    snu.solve_nse(**soldict)


if __name__ == '__main__':
    nonltrt = True
    # # ## baby
    # testit(problem='cylinderwake', N=1, Re=30, t0=0.0, tE=.1, Nts=50,
    #        scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## light
    testit(problem='cylinderwake', N=2, Re=80, t0=0.0, tE=1., Nts=512,
           scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## medium
    # testit(problem='cylinderwake', N=2, Re=100, t0=0.0, tE=2., Nts=4*512,
    #        scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## hard
    # testit(problem='cylinderwake', N=3, Re=150, t0=0.0, tE=2., Nts=8*512,
    #        scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## 3D
    # testit(problem='cylinderwake3D', N=2, Re=50, t0=0.0, tE=2., Nts=512,
    #        scheme='CR', ParaviewOutput=True)
