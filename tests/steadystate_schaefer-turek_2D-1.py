import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

geodata = 'mesh/karman2D-rotcyl-bm_geo_cntrlbc.json'
proutdir = 'results/'


def testit(problem=None, nu=None, charvel=None, Re=None,
           meshlvl=1,
           rho=1.,
           t0=0.0, tE=1.0, Nts=1e2+1, ParaviewOutput=False, scheme='TH'):

    meshfile = 'mesh/karman2D-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/karman2D-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='cylinder_rot', nu=nu, bccontrol=False,
                          charvel=charvel,
                          scheme=scheme, mergerhs=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))
    ddir = 'data/'
    data_prfx = problem + '{4}_mesh{0}_Re{1}_Nts{2}_tE{3}'.\
        format(meshlvl, femp['Re'], Nts, tE, scheme)

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    # ## Parameters for the benchmark values
    Um = charvel  # (we alread scale the inflow parabola accordingly)
    L = femp['charlen']  # characteristic length
    NP, NV = stokesmatsc['J'].shape
    print('NV + NP : {0} + {1} = {2}'.format(NV, NP, NV+NP))

    def rotcont(t, vel):
        return 0.

    dircntdict = dict(diricontbcinds=[femp['mvwbcinds']],
                      diricontbcvals=[femp['mvwbcvals']],
                      diricontfuncs=[rotcont])

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(dircntdict)
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=meshlvl, nu=nu,
                   # start_ssstokes=True,
                   verbose=True,
                   return_vp=True,
                   iniv=0*rhsd['fv'],
                   get_datastring=None,
                   comp_nonl_semexp=True,
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
    realpss = rho*dynpss  # Um**2*rho*dynpss
    realvss = vss  # Um*vss
    getld = dnsps.LiftDragSurfForce(V=femp['V'], nu=nu,
                                    ldds=femp['liftdragds'])
    clift, cdrag = getld.evaliftdragforce(u=realvss, p=realpss)
    cdclfac = 2./(rho*L*Um**2)
    print('Cl: {0}'.format(cdclfac*clift))
    print('Cd: {0}'.format(cdclfac*cdrag))
    import dolfin
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

    # snu.solve_nse(**soldict)
    # print('for plots check \nparaview ' + proutdir + 'vel___timestep.pvd')
    # print('or \nparaview ' + proutdir + 'p___timestep.pvd')


if __name__ == '__main__':
    meshlvl = 3
    nu = 1e-3
    rho = 1.
    charvel = .2
    scheme = 'CR'
    scheme = 'TH'

    testit(problem='gen_bccont', nu=nu, charvel=charvel,
           rho=rho,
           t0=0.0, tE=2., Nts=512, meshlvl=meshlvl,
           scheme=scheme, ParaviewOutput=True)
