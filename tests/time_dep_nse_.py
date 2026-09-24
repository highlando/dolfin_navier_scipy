import logging

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

from rich.logging import RichHandler

# dolfin.parameters.linear_algebra_backend = 'uBLAS'

# krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-2,
#                                              'convstatsl': [],
#                                              'maxiter': 200})
krylovdict = {}

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

def testit(problem='drivencavity', N=None, nu=None, Re=None, Nts=1e3,
           ParaviewOutput=False, nsects=1, addfullsweep=False,
           tE=1.0, scheme=None, symmetricgrad=True):

    nnewtsteps = 9  # n nwtn stps for vel comp
    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd = dnsps.\
        get_sysmats(problem=problem, Re=Re, nu=nu, scheme=scheme,
                    meshparams=dict(refinement_level=N), mergerhs=True,
                    gradvsymmtrc=symmetricgrad)
    proutdir = 'results/'
    ddir = 'data/'
    data_prfx = problem + '_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE)

    soldict = {}
    soldict.update(stokesmatsc)  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(rhsd)
    symgstr = 'symg' if symmetricgrad else ''
    soldict.update(N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   nsects=nsects, addfullsweep=addfullsweep,
                   start_ssstokes=True,
                   get_datastring=None,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput, prvoutpnts=100,
                   vel_pcrd_stps=1,
                   clearprvdata=True,
                   vfileprfx=proutdir+f'vel{symgstr}_{scheme}_',
                   pfileprfx=proutdir+f'p{symgstr}_{scheme}_')

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)
    # print krylovdict['krpslvprms']['convstatsl']


if __name__ == '__main__':
    # scme = 0
    # schemel = ['CR', 'TH']
    # scheme = schemel[scme]
    # testit(problem='cylinderwake', N=2, Re=40, Nts=25, tE=.1)
    symg = True
    symg = False
    testit(problem='cylinderwake', N=2, Re=30, Nts=int(2**10), tE=2.,
           symmetricgrad=symg, ParaviewOutput=True)
    # testit(N=40, Re=1e3, Nts=.5e2, tE=.5, ParaviewOutput=True, scheme=scheme)
    # testit(problem='cylinderwake', N=2, Re=80, Nts=512, tE=1.,
    #        scheme='TH', nsects=4, ParaviewOutput=True)
    # testit(problem='cylinderwake', N=4, Re=80, Nts=1000, tE=1.,
    #        ParaviewOutput=True, scheme='CR')
