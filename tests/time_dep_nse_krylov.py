import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

krylovdict = dict(krylov='Gmres', krpslvprms={'tol': 1e-3,
                                              # 'convstatsl': [],
                                              # 'krylovini': 'upd',
                                              'maxiter': 800})
# krylovdict = {}


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None, Nts=1e3,
           ParaviewOutput=False, tE=1.0):

    nnewtsteps = 4  # n nwtn stps for vel comp
    npcrdsteps = 0  # n picard steps
    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd = dnsps.\
        get_sysmats(problem=problem, nu=nu, mergerhs=True,
                    meshparams=dict(refinement_level=N))
    proutdir = 'results/'
    ddir = 'data/'
    data_prfx = problem + '_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   vel_pcrd_stps=npcrdsteps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   start_ssstokes=True,
                   get_datastring=None,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   clearprvdata=True,
                   comp_nonl_semexp=True,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

    soldict.update(krylovdict)  # if we wanna use an iterative solver

    snu.solve_nse(**soldict)
    print(len(krylovdict['krpslvprms']['convstatsl']))


if __name__ == '__main__':
    # testit(N=15, nu=1e-2)
    testit(problem='cylinderwake', N=1, Re=40, Nts=256, tE=.5,
           ParaviewOutput=True)
