import numpy as np
import matplotlib.pyplot as plt
import json

import dolfin

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

from dolfin_navier_scipy.residual_checks import get_imex_res
# from dolfin_navier_scipy.residual_checks import get_steady_state_res


def twod_simu(nu=None, charvel=None, rho=1., rhosolid=10., meshparams=None,
              inirot=None, inivfun=None,
              t0=0.0, tE=.1, Nts=1e2+1,
              start_steadystate=False, ininu=None,
              plotplease=False, proutdir='paraviewplots/',
              return_final_vp=False, ParaviewOutput=False, scheme='TH'):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', nu=nu, bccontrol=False,
                          charvel=charvel, scheme=scheme, mergerhs=True,
                          meshparams=meshparams)
    # dnsps.get_sysmats(problem='cylinder_rot', nu=nu, bccontrol=False,
    #                   charvel=charvel, scheme=scheme, mergerhs=True,
    #                   meshparams=meshparams)

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    NP, NV = stokesmatsc['J'].shape
    print('NV + NP : {0} + {1} = {2}'.format(NV, NP, NV+NP))

    # function of ones at the lift/drag boundary
    phionevec = np.zeros((femp['V'].dim(), 1))
    phionevec[femp['mvwbcinds'], :] = 1.
    phione = dolfin.Function(femp['V'])
    phione.vector().set_local(phionevec)
    pickx = dolfin.as_matrix([[1., 0.], [0., 0.]])
    picky = dolfin.as_matrix([[0., 0.], [0., 1.]])
    pox = pickx*phione
    poy = picky*phione

    # function of the tangential vector at the lift/drag boundary
    phitwovec = np.zeros((femp['V'].dim(), 1))
    phitwovec[femp['mvwbcinds'], 0] = femp['mvwbcvals']
    phitwo = dolfin.Function(femp['V'])
    phitwo.vector().set_local(phitwovec)

    # getld = dnsps.LiftDragSurfForce(V=femp['V'], nu=nu,
    #                                 phione=phione, phitwo=phitwo,
    #                                 outflowds=femp['outflowds'],
    #                                 ldds=femp['liftdragds'])

    # L = femp['charlen']  # characteristic length = 2*Radius

    a_1 = dolfin.Point(0.15, 0.2)
    a_2 = dolfin.Point(0.25, 0.2)

    reschkdict = dict(V=femp['V'], gradvsymmtrc=True,
                      outflowds=femp['outflowds'], nu=nu)
    euleres = get_imex_res(explscheme='eule', **reschkdict)
    heunres = get_imex_res(explscheme='heun', **reschkdict)
    abtwres = get_imex_res(explscheme='abtw', **reschkdict)
    # ststres = get_steady_state_res(**reschkdict)

    def record_ldt(t, vel=None, p=None, memory={}, mode='abtwo'):

        rotval = 0.
        if mode == 'stokes':
            memory.update(dict(lastt=t))
            return rotval, memory

        if mode == 'init':
            memory.update(dict(lastt=t))
            return rotval, memory

        vfun, pfun = dts.expand_vp_dolfunc(vc=vel, pc=p, **femp)

        if mode == 'heunpred' or mode == 'heuncorr':
            curdt = t - memory['lastt']
            if mode == 'heunpred':
                memory.update(dict(lastv=vel))
                pass

            elif mode == 'heuncorr':
                lvfun = dts.expand_vp_dolfunc(vc=memory['lastv'],
                                              **femp)[0]
                trqe = euleres(vfun, pfun, curdt, lastvel=lvfun, phi=phitwo)
                lift = euleres(vfun, pfun, curdt, lastvel=lvfun, phi=poy)
                drag = euleres(vfun, pfun, curdt, lastvel=lvfun, phi=pox)
                memory.update(dict(lastt=t, lastdt=curdt, heunpred=vel))

                memory['trqs'].append(trqe)
                memory['lfts'].append(lift)
                memory['drgs'].append(drag)
                memory['tims'].append(t)

        elif mode == 'abtwo':

            lvfun = dts.expand_vp_dolfunc(vc=memory['lastv'], **femp)[0]
            curdt = t - memory['lastt']

            try:
                ovfn = dts.expand_vp_dolfunc(vc=memory['lastlastv'], **femp)[0]
                modres = abtwres
            except KeyError:  # no lastlastv yet -- we can check the Heun res
                ovfn = dts.expand_vp_dolfunc(vc=memory['heunpred'], **femp)[0]
                modres = heunres

            trqe = modres(vfun, pfun, curdt, lastvel=lvfun, othervel=ovfn,
                          phi=phitwo)
            lift = modres(vfun, pfun, curdt, lastvel=lvfun, othervel=ovfn,
                          phi=poy)
            drag = modres(vfun, pfun, curdt, lastvel=lvfun, othervel=ovfn,
                          phi=pox)

            memory.update(dict(lastlastv=np.copy(memory['lastv'])))
            memory.update(dict(lastv=vel))

            memory['trqs'].append(trqe)
            memory['lfts'].append(lift)
            memory['drgs'].append(drag)
            memory['tims'].append(t)
            memory.update(dict(lastt=t, lastdt=curdt))

        deltap = pfun(a_1) - pfun(a_2)
        memory['dtps'].append(deltap)
        return rotval, memory

    rotcondict = dict(lastt=None,
                      trqs=[], omegs=[], lfts=[], drgs=[], dtps=[], tims=[],
                      lastdt=None)

    dircntdict = dict(diricontbcinds=[femp['mvwbcinds']],
                      diricontbcvals=[femp['mvwbcvals']],
                      diricontfuncs=[record_ldt],
                      diricontfuncmems=[rotcondict])

    soldict = {}
    soldict.update(stokesmatsc)  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(dircntdict)
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   verbose=True,
                   vel_pcrd_stps=5,
                   return_vp=True,
                   treat_nonl_explicit=True,
                   no_data_caching=True,
                   return_final_vp=return_final_vp,
                   dbcinds=femp['dbcinds'], dbcvals=femp['dbcvals'],
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

#
    if inivfun is None:
        if start_steadystate:
            if ininu is not None:
                inifemp, inistokesmatsc, inirhsd = \
                    dnsps.get_sysmats(problem='cylinder_rot', nu=ininu,
                                      bccontrol=False,
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
            soldict.update(vel_nwtn_tol=1e-3)
            vp_ss_nse = snu.solve_steadystate_nse(**soldict)
            soldict.update(dict(iniv=vp_ss_nse[0]))
        else:
            soldict.update(start_ssstokes=True)
    else:
        inivvec = (inivfun.vector().get_local()).reshape((femp['V'].dim(), 1))
        soldict.update(dict(iniv=inivvec))

    finalvp = snu.solve_nse(**soldict)
    if ParaviewOutput:
        print('for plots check \nparaview ' + proutdir + 'vel___timestep.pvd')
        print('or \nparaview ' + proutdir + 'p___timestep.pvd')

    resdict = rotcondict

    nnz = 2*stokesmatsc['J'].nnz + stokesmatsc['A'].nnz

    resdict.update(dict(nvnp=[NV, NP], nnz=nnz))
    if return_final_vp:
        return rotcondict, finalvp

    return rotcondict


if __name__ == '__main__':
    nu = 1e-3
    rho = 1.
    # rhosolid = 10.

    geodata = 'mesh/karman2D-rotcyl-bm_geo_cntrlbc.json'

    scheme = 'TH'
    charvel = 1.
    baseNts = 2*2048
    scaletest = 5.
    L, Um = 0.1, charvel
    warmstart = True
    restart = False

    tau = 1./baseNts
    lgtwo = np.log(tau)/np.log(2)
    meshlvl = 1

    infostr = ('\nt0  = {0}'.format(0.) +
               '\ntE  = {0}'.format(scaletest*1.) +
               '\ntau = 2**({0:.0f})'.format(lgtwo) +
               '\nmesh= {0}'.format(meshlvl))

    print(infostr)

    meshfile = 'mesh/karman2D-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/karman2D-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    meshparams = dict(strtomeshfile=meshfile,
                      movingwallcntrl=True,
                      strtophysicalregions=physregs, strtobcsobs=geodata)
    wrmstrtstr = 'results/tdp-ml{0}-forwarmstart'.format(meshlvl)

    tinfo = {}
    resdct, finalvp = twod_simu(nu=nu, charvel=charvel,
                                meshparams=meshparams,
                                rho=rho,  # rhosolid=rhosolid,
                                t0=0., tE=scaletest*1.,
                                Nts=np.int(scaletest*baseNts),
                                # inivfun=inivfun, inirot=inirot,
                                return_final_vp=True,
                                plotplease=True, ParaviewOutput=False,
                                proutdir='paraviewplots/')
    jsfile = open('results/tdp-ml{0}'.format(meshlvl) + '.json',
                  mode='w')
    resdct['lastv'] = resdct['lastv'].tolist()
    resdct['lastlastv'] = resdct['lastlastv'].tolist()
    resdct['heunpred'] = resdct['heunpred'].tolist()
    resdct.update(dict(meshlvl=meshlvl))
    drgfac = 2/(Um**2*rho*L)
    lftfac = 2/(Um**2*rho*L)
    lfts = lftfac*np.array(resdct['lfts'])
    drgs = drgfac*np.array(resdct['drgs'])

    plt.figure(1)
    plt.plot(lfts[10:])
    plt.figure(2)
    plt.plot(drgs[10:])
    plt.show()
    jsfile.write(json.dumps(resdct))
    print('results dumped to \n', jsfile.name)
    jsfile.close()
