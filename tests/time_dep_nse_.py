import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2,
#                                              'convstatsl': [],
#                                              'maxiter': 200})
krylovdict = {}


def testit(problem='drivencavity', N=None, nu=None, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0, scheme=None):

    nnewtsteps = 9  # n nwtn stps for vel comp
    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd_vfrc, \
        rhsd_stbc, data_prfx, ddir, proutdir \
        = dnsps.get_sysmats(problem=problem, N=N, Re=Re, nu=nu, scheme=scheme)

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
                   vel_pcrd_stps=1,
                   clearprvdata=True,
                   vfileprfx=proutdir+'vel_{0}_'.format(scheme),
                   pfileprfx=proutdir+'p_{0}_'.format(scheme))

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)
    # print krylovdict['krpslvprms']['convstatsl']


if __name__ == '__main__':
    scme = 1
    schemel = ['CR', 'TH']
    scheme = schemel[scme]
    # testit(N=40, Re=1e3, Nts=.5e2, tE=.5, ParaviewOutput=True, scheme=scheme)
    testit(problem='cylinderwake', N=2, Re=50, Nts=128, tE=1.,
           ParaviewOutput=True)
