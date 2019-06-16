# import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, nonltrt=None,
           t0=0.0, tE=1.0, Nts=1e2+1, ParaviewOutput=False, scheme='TH'):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem=problem, Re=Re, nu=nu, scheme=scheme,
                          meshparams=dict(refinement_level=N), mergerhs=True)

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=N, nu=nu,
                   get_datastring=None,
                   treat_nonl_explct=nonltrt,
                   ret_vp_dict=True)

    soldict.update(dict(start_ssstokes=True))
    snu.solve_nse(**soldict)


if __name__ == '__main__':
    nonltrt = True
    Nts = 3
    # # ## baby
    # testit(problem='cylinderwake', N=1, Re=30, t0=0.0, tE=.1, Nts=3,
    #        scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## light
    # testit(problem='cylinderwake', N=2, Re=80, t0=0.0, tE=1., Nts=512,
    #        scheme='CR', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## medium
    testit(problem='cylinderwake', N=2, Re=100, t0=0.0, tE=2., Nts=4*512,
           scheme='TH', nonltrt=nonltrt)
    # # ## hard
    # testit(problem='cylinderwake', N=3, Re=150, t0=0.0, tE=2., Nts=8*512,
    #        scheme='TH', ParaviewOutput=True, nonltrt=nonltrt)
    # # ## 3D
    # testit(problem='cylinderwake3D', N=2, Re=50, t0=0.0, tE=2., Nts=512,
    #        scheme='CR', ParaviewOutput=True)
