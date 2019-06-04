import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

# dolfin.parameters.linear_algebra_backend = 'uBLAS'


def testit(problem='cylinderwake', N=None, nu=None, Re=None,
           nnwtnstps=9, npcrdstps=5, palpha=1e-5):

    vel_nwtn_tol = 1e-14
    # prefix for data files
    data_prfx = problem
    # dir to store data
    ddir = 'data/'
    # paraview output
    ParaviewOutput = True
    proutdir = 'results/'

    femp, stokesmatsc, rhsd = dnsps.\
        get_sysmats(problem=problem, Re=Re, nu=nu, scheme='TH', mergerhs=True,
                    bccontrol=True, meshparams=dict(refinement_level=N))
    import scipy.sparse.linalg as spsla
    print('get expmats: |A| = {0}'.format(spsla.norm(stokesmatsc['A'])))
    print('get expmats: |Arob| = {0}'.format(spsla.norm(stokesmatsc['Arob'])))

    stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
    b_mat = 1./palpha*stokesmatsc['Brob']
    brhs = 1*(1.5*b_mat[:, :1] - 1.5*b_mat[:, 1:])

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs

    soldict.update(fv=rhsd['fv']+brhs, fp=rhsd['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnwtnstps,
                   vel_pcrd_stps=npcrdstps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   ddir=ddir, get_datastring=None,
                   clearprvdata=True,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    snu.solve_steadystate_nse(**soldict)
    print('plots go to : ' + proutdir + 'vel___steadystates.pvd')


if __name__ == '__main__':
    # testit(N=25, nu=3e-4)
    # testit(problem='cylinderwake', N=3, nu=2e-3)
    # testit(problem='drivencavity', N=25, Re=500)
    testit(problem='cylinderwake', N=2, Re=40, nnwtnstps=10, npcrdstps=5)
