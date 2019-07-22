# import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.dolfin_to_sparrays as dts

from dolfin_navier_scipy.residual_checks import get_imex_res


def testit(problem='gen_bccont', nu=1e-3, charvel=0.2,
           t0=0.0, tE=.1, Nts=2,
           rho=1., meshlvl=1, meshparams=1, nonltrt=True,
           scheme=None):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem=problem, nu=nu, bccontrol=False,
                          charvel=charvel, scheme=scheme, mergerhs=True,
                          meshparams=meshparams)

    # setting some parameters

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   treat_nonl_explct=nonltrt,
                   return_vp_dict=True,
                   no_data_caching=True,
                   start_ssstokes=True)

    vpdct = snu.solve_nse(**soldict)
    M, A, JT = stokesmatsc['M'], stokesmatsc['A'], stokesmatsc['JT']
    fv = rhsd['fv']
    V, invinds = femp['V'], femp['invinds']
    dt = (tE-t0)/Nts
    tm = (tE-t0)/2

    reschkdict = dict(V=V, nu=nu,
                      gradvsymmtrc=True, outflowds=femp['outflowds'])
    euleres = get_imex_res(explscheme='eule', **reschkdict)
    heunres = get_imex_res(explscheme='heun', **reschkdict)
    crnires = get_imex_res(explscheme='abtw', **reschkdict)

    # the initial value
    inivwbcs = vpdct[t0]['v']
    iniv = inivwbcs[invinds]
    iniconvvec = dts.get_convvec(V=V, u0_vec=inivwbcs, invinds=invinds)
    inivelfun = dts.expand_vp_dolfunc(vc=inivwbcs, **femp)[0]

    # the Heun prediction step
    cneevwbcs = vpdct[(tm, 'heunpred')]['v']
    cneev = cneevwbcs[invinds]
    cneep = vpdct[(tm, 'heunpred')]['p']

    # the Heun step
    cnhevwbcs = vpdct[tm]['v']
    cnhev = cnhevwbcs[invinds]
    cnhep = vpdct[tm]['p']
    hpconvvec = dts.get_convvec(V=V, u0_vec=cneevwbcs, invinds=invinds)
    hpvelfun = dts.expand_vp_dolfunc(vc=cneevwbcs, **femp)[0]

    # the AB2 step
    cnabvwbcs = vpdct[tE]['v']
    cnabv = cnabvwbcs[invinds]
    cnabp = vpdct[tE]['p']
    hcconvvec = dts.get_convvec(V=V, u0_vec=cnhevwbcs, invinds=invinds)
    hcvelfun = dts.expand_vp_dolfunc(vc=cnhevwbcs, **femp)[0]

    print('Heun-Prediction: one step of Euler')
    resvec = (1./dt*M*(cneev-iniv) + .5*A*(iniv+cneev)
              + iniconvvec - JT*cneep - fv)
    print('Scipy residual: ', np.linalg.norm(resvec))
    curv, curp = dts.expand_vp_dolfunc(vc=cneevwbcs, pc=cneep, **femp)
    res = euleres(curv, curp, dt, lastvel=inivelfun)
    print('dolfin residua: ', np.linalg.norm(res.get_local()[invinds]))

    print('\nHeun-Step:')
    heunrhs = M*iniv - .5*dt*(A*iniv + iniconvvec+hpconvvec) + dt*fv
    matvp = M*cnhev + .5*dt*A*cnhev - dt*JT*cnhep
    print('Scipy residual: ', np.linalg.norm(matvp - heunrhs))
    # import ipdb; ipdb.set_trace()
    curv, curp = dts.expand_vp_dolfunc(vc=cnhevwbcs, pc=cnhep, **femp)
    heunres = heunres(curv, curp, dt, lastvel=inivelfun, othervel=hpvelfun)
    print('dolfin residua: ', np.linalg.norm(heunres.get_local()[invinds]))

    print('\nAB2-Step:')
    abtrhs = M*cnhev - .5*dt*(A*cnhev + -iniconvvec + 3.*hcconvvec) + dt*fv
    matvp = M*cnabv + .5*dt*A*cnabv - dt*JT*cnabp
    print('Scipy residual: ', np.linalg.norm(matvp - abtrhs))
    # import ipdb; ipdb.set_trace()
    curv, curp = dts.expand_vp_dolfunc(vc=cnabvwbcs, pc=cnabp, **femp)
    crnires = crnires(curv, curp, dt, lastvel=hcvelfun, othervel=inivelfun)
    print('dolfin residua: ', np.linalg.norm(crnires.get_local()[invinds]))


if __name__ == '__main__':

    meshlvl = 1
    geodata = 'mesh/karman2D-rotcyl-bm_geo_cntrlbc.json'
    meshfile = 'mesh/karman2D-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/karman2D-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    meshparams = dict(strtomeshfile=meshfile,
                      movingwallcntrl=False,
                      strtophysicalregions=physregs, strtobcsobs=geodata)
    nonltrt = True
    Nts = 2
    scheme = 'TH'
    testit(problem='gen_bccont', nu=1e-3, charvel=0.2,
           t0=0.0, tE=.1, Nts=Nts, rho=1.,
           meshparams=meshparams, nonltrt=nonltrt,
           scheme=scheme)
