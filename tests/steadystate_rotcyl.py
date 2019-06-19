import numpy as np
import scipy.optimize as sco

import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.data_output_utils as dou

from dolfin_navier_scipy.residual_checks import get_steady_state_res

geodata = 'mesh/karman2D-rotcyl-bm_geo_cntrlbc.json'
proutdir = 'results/'
rhosolid = 10.


def testit(problem=None, nu=None, ininu=None, charvel=None,
           meshlvl=1, rho=1., ParaviewOutput=False, scheme='TH'):

    meshfile = 'mesh/karman2D-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/karman2D-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    meshparams = dict(strtomeshfile=meshfile,
                      strtophysicalregions=physregs, strtobcsobs=geodata)
    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem=problem, nu=nu, bccontrol=False,
                          charvel=charvel, scheme=scheme, mergerhs=True,
                          meshparams=meshparams)
    soldict = {}
    soldict.update(stokesmatsc)  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs

    ddir = 'data/'
    data_prfx = problem + '{2}_mesh{0}_Re{1}'.\
        format(meshlvl, femp['Re'], scheme)
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=meshlvl, nu=nu,
                   verbose=True,
                   vel_pcrd_stps=0,
                   vel_nwtn_tol=1e-10,
                   vel_nwtn_stps=10,
                   return_vp=True,
                   get_datastring=None,
                   dbcinds=femp['dbcinds'], dbcvals=femp['dbcvals'],
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')
    L = femp['charlen']  # characteristic length

    phionevec = np.zeros((femp['V'].dim(), 1))
    phionevec[femp['mvwbcinds'], :] = 1.
    phione = dolfin.Function(femp['V'])
    phione.vector().set_local(phionevec)
    pickx = dolfin.as_matrix([[1., 0.], [0., 0.]])
    picky = dolfin.as_matrix([[0., 0.], [0., 1.]])
    pox = pickx*phione
    poy = picky*phione

    phitwovec = np.zeros((femp['V'].dim(), 1))
    phitwovec[femp['mvwbcinds'], 0] = femp['mvwbcvals']
    phitwo = dolfin.Function(femp['V'])
    phitwo.vector().set_local(phitwovec)
    if ParaviewOutput:
        phifile = dolfin.File('results/phione.pvd')
        phifile << phitwo

    # getld = dnsps.LiftDragSurfForce(V=femp['V'], nu=nu,
    #                                 phione=phione, phitwo=phitwo,
    #                                 outflowds=femp['outflowds'],
    #                                 ldds=femp['liftdragds'])

    steady_state_res = \
        get_steady_state_res(V=femp['V'], gradvsymmtrc=True,
                             outflowds=femp['outflowds'], nu=nu)

    def comptorque(rotval, thingdict=None, returnitall=False):

        def rotcont(t, vel=None, p=None, memory={}):
            return rotval, memory

        rotcondict = {}
        dircntdict = dict(diricontbcinds=[femp['mvwbcinds']],
                          diricontbcvals=[femp['mvwbcvals']],
                          diricontfuncs=[rotcont],
                          diricontfuncmems=[rotcondict])
        soldict.update(dircntdict)
        soldict.update(dict(vel_start_nwtn=thingdict['vel_start_nwtn']))
        if ininu is not None and thingdict['vel_start_nwtn'] is None:
            inifemp, inistokesmatsc, inirhsd = \
                dnsps.get_sysmats(problem=problem, nu=ininu, bccontrol=False,
                                  charvel=charvel, scheme=scheme,
                                  mergerhs=True, meshparams=meshparams)
            soldict.update(inistokesmatsc)
            soldict.update(inifemp)
            soldict.update(fv=inirhsd['fv'], fp=inirhsd['fp'])
            vp_ss_nse = snu.solve_steadystate_nse(**soldict)
            soldict.update(dict(vel_start_nwtn=vp_ss_nse[0]))
            soldict.update(stokesmatsc)
            soldict.update(femp)
            soldict.update(fv=rhsd['fv'], fp=rhsd['fp'])

        vp_ss_nse = snu.solve_steadystate_nse(**soldict)
        thingdict.update(dict(vel_start_nwtn=vp_ss_nse[0]))

        if returnitall:
            vfun, pfun = dts.\
                expand_vp_dolfunc(vc=vp_ss_nse[0], pc=vp_ss_nse[1],
                                  V=femp['V'], Q=femp['Q'])
            # lift, drag = getld.evaliftdragforce(u=vfun, p=pfun)

            drag = steady_state_res(vfun, pfun, phi=pox)
            lift = steady_state_res(vfun, pfun, phi=poy)
            # phionex = phione.sub(0)

            trqe = steady_state_res(vfun, pfun, phi=phitwo)
            # trqe = getld.evatorqueSphere2D(u=vfun, p=pfun)
            a_1 = dolfin.Point(0.15, 0.2)
            a_2 = dolfin.Point(0.25, 0.2)
            pdiff = rho*pfun(a_2) - rho*pfun(a_1)
            return trqe, lift, drag, pdiff
        else:
            vfun, pfun = dts.\
                expand_vp_dolfunc(vc=vp_ss_nse[0], pc=vp_ss_nse[1],
                                  V=femp['V'], Q=femp['Q'])
            # trqe = getld.evatorqueSphere2D(u=vfun, p=pfun)
            trqe = steady_state_res(vfun, pfun, phi=phitwo)

            print('omeg: {0:.3e} -- trqe: {1:.3e}'.format(rotval, trqe))
            return np.abs(trqe)

    Um = charvel
    thingdict = dict(vel_start_nwtn=None)

    testrot = 0.
    trqe, lift, drag, pdif = comptorque(testrot, thingdict, returnitall=True)
    print('\n\n# ## Nonrotating Cylinder ')

    cdclfac = 2./(rho*L*Um**2)
    trqefac = 4/(Um**2*rho*L**2)
    print('Cl: {0:.9f}'.format(cdclfac*lift))
    print('Cd: {0:.9f}'.format(cdclfac*drag))
    print('Ct: {0:.5e}'.format(trqefac*trqe))
    print('Delta P: {0:.9f}'.format(pdif))

    if charvel == 0.2:
        print('\n cp. values from Schaefer/Turek as in')
        print('www.featflow.de/en/benchmarks/cfdbenchmarking/flow/' +
              'dfg_benchmark1_re20.html:')
        print('Cl: {0:.8f}'.format(0.010618948146))
        print('Cd: {0:.8f}'.format(5.57953523384))
        print('Delta P: {0:.8f}'.format(0.11752016697))

    print('\n\n# ## Rotating Cylinder -- optimizing rotation for zero torque')
    tinfo = {}
    with dou.Timer(timerinfo=tinfo):
        res = sco.minimize_scalar(comptorque, args=(thingdict),
                                  options={'maxiter': 80}, tol=1e-13)
    trqe, lift, drag, pdiff = comptorque(res['x'], thingdict, returnitall=True)

    print('\n# ## Rotating Cylinder -- optimized rotation for zero torque')
    print('omega*: {0:.8f}'.format(res['x']*L/(2*Um)))
    print('Cl: {0:.8f}'.format(cdclfac*lift))
    print('Cd: {0:.8f}'.format(cdclfac*drag))
    print('Ct: {0:.4e}'.format(trqefac*trqe))
    print('Delta P: {0:.8f}'.format(pdif))
    if charvel == 0.2:
        print('\n cp. values from Richter et. al')
        print('omega*: {0}'.format(0.00126293))
        print('Cl: {0}'.format(0.0047141))
        print('Cd: {0}'.format(5.579558))
        print('Delta P: {0}'.format(0.117520))

if __name__ == '__main__':
    setup = 'rot2d-2'
    setup = 'rot2d-1'
    nu = 1e-3
    rho = 1.
    scheme = 'CR'
    scheme = 'TH'
    problem = 'cylinder_rot'

    if setup == 'rot2d-1':
        meshlvllist = [3]
        charvel = .2
        ininu = None

    elif setup == 'rot2d-2':
        meshlvllist = [4]  # , 5]
        charvel = 1.
        ininu = 1.25*nu

    for meshlvl in meshlvllist:
        resdct = testit(problem=problem, nu=nu, charvel=charvel, rho=rho,
                        ininu=ininu, meshlvl=meshlvl, scheme=scheme,
                        ParaviewOutput=True)
