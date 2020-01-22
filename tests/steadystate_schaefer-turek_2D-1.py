import numpy as np

import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

from dolfin_navier_scipy.residual_checks import get_steady_state_res

geodata = 'mesh/karman2D-rotcyl-bm_geo_cntrlbc.json'
proutdir = 'results/'


def testit(problem=None, nu=None, charvel=None, Re=None,
           meshlvl=1, gradvsymmtrc=True,
           rho=1.,
           ParaviewOutput=False, scheme='TH'):

    meshfile = 'mesh/karman2D-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/karman2D-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem=problem, nu=nu,
                          charvel=charvel, gradvsymmtrc=gradvsymmtrc,
                          scheme=scheme, mergerhs=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          movingwallcntrl=False,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))

    ddir = 'data/'
    data_prfx = problem+'{2}_mesh{0}_Re{1}'.format(meshlvl, femp['Re'], scheme)

    # ## Parameters for the benchmark values
    Um = charvel  # (we alread scale the inflow parabola accordingly)
    L = femp['charlen']  # characteristic length
    NP, NV = stokesmatsc['J'].shape
    print('NV + NP : {0} + {1} = {2}'.format(NV, NP, NV+NP))

    soldict = stokesmatsc  # containing A, J, JT
    # soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(invinds=femp['invinds'], V=femp['V'], Q=femp['Q'])
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=meshlvl, nu=nu,
                   verbose=True,
                   return_vp=True,
                   get_datastring=None,
                   dbcinds=femp['dbcinds'], dbcvals=femp['dbcvals'],
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    vp_ss_nse = snu.solve_steadystate_nse(**soldict)
    vss, dynpss = dts.expand_vp_dolfunc(vc=vp_ss_nse[0], pc=vp_ss_nse[1],
                                        **femp)
    steady_state_res = \
        get_steady_state_res(V=femp['V'], gradvsymmtrc=True,
                             outflowds=femp['outflowds'], nu=nu)

    res = steady_state_res(vss, rho*dynpss)
    auxvec = np.zeros((femp['V'].dim(), ))
    invinds = femp['invinds']
    auxvec[invinds] = res.get_local()[invinds]
    print('two norm of the res: {0}'.format(np.linalg.norm(auxvec)))

    phionevec = np.zeros((femp['V'].dim(), 1))
    phionevec[femp['ldsbcinds'], :] = 1.
    phione = dolfin.Function(femp['V'])
    phione.vector().set_local(phionevec)
    pickx = dolfin.as_matrix([[1., 0.], [0., 0.]])
    picky = dolfin.as_matrix([[0., 0.], [0., 1.]])
    pox = pickx*phione
    poy = picky*phione
    drag = steady_state_res(vss, rho*dynpss, phi=pox)
    lift = steady_state_res(vss, rho*dynpss, phi=poy)
    cdclfac = 2./(rho*L*Um**2)

    print('Computed via testing the residual: ')
    print('Cl: {0}'.format(cdclfac*lift))
    print('Cd: {0}'.format(cdclfac*drag))

    phionevec = np.zeros((femp['V'].dim(), 1))
    phionevec[femp['ldsbcinds'], :] = 1.
    phione = dolfin.Function(femp['V'])
    phione.vector().set_local(phionevec)
    # phionex = phione.sub(0)

    print('Computed via `dnsps.LiftDragSurfForce`:')
    realpss = rho*dynpss  # Um**2*rho*dynpss
    realvss = vss  # Um*vss
    getld = dnsps.LiftDragSurfForce(V=femp['V'], nu=nu,
                                    ldds=femp['liftdragds'],
                                    outflowds=femp['outflowds'],
                                    phione=phione)
    clift, cdrag = getld.evaliftdragforce(u=realvss, p=realpss)
    print('Cl: {0}'.format(cdclfac*clift))
    print('Cd: {0}'.format(cdclfac*cdrag))

    a_1 = dolfin.Point(0.15, 0.2)
    a_2 = dolfin.Point(0.25, 0.2)
    pdiff = realpss(a_1) - realpss(a_2)
    print('Delta P: {0}'.format(pdiff))

    print('\n values from Schaefer/Turek as in')
    print('www.featflow.de/en/benchmarks/cfdbenchmarking/flow/' +
          'dfg_benchmark1_re20.html:')
    print('Cl: {0}'.format(0.010618948146))
    print('Cd: {0}'.format(5.57953523384))
    print('Delta P: {0}'.format(0.11752016697))


if __name__ == '__main__':
    meshlvl = 1
    nu = 1e-3
    rho = 1.
    charvel = .2

    # scheme = 'CR'
    # testit(problem='gen_bccont', nu=nu, charvel=charvel,
    #        rho=rho, meshlvl=meshlvl, gradvsymmtrc=False,
    #        scheme=scheme, ParaviewOutput=True)

    scheme = 'TH'
    testit(problem='gen_bccont', nu=nu, charvel=charvel,
           rho=rho, meshlvl=meshlvl,  # gradvsymmtrc=False,
           scheme=scheme, ParaviewOutput=True)
