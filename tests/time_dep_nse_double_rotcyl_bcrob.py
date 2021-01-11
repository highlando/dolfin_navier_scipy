import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

geodata = 'mesh/2D-double-rotcyl_geo_cntrlbc_rotcntrl.json'
proutdir = 'results/'


def testit(charvel=None, Re=None,
           meshlvl=1, gradvsymmtrc=True,
           rho=1., tE=None, Nts=None, zerocontrol=False,
           ParaviewOutput=False, scheme='TH'):

    meshfile = 'mesh/2D-double-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/2D-double-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re,
                          gradvsymmtrc=gradvsymmtrc,
                          scheme=scheme, mergerhs=True,
                          bccontrol=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          movingwallcntrl=False,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))
    proutdir = 'results/'
    ddir = 'data/'
    data_prfx = 'doublecylrobbc_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(meshlvl, femp['Re'], Nts, tE)

    dolfin.plot(femp['mesh'])

    palpha = 1e-5
    stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
    if zerocontrol:
        Brob = 0.*1./palpha*stokesmatsc['Brob']
    else:
        Brob = 1./palpha*stokesmatsc['Brob']

    def fv_tmdp(time=0, v=None, **kw):
        # return np.sin(time)*(Brob[:, :1] - Brob[:, 1:]), None
        # return 1*(1-np.exp(-time))*(Brob[:, :1] + Brob[:, 1:])
        return np.sin(time/tE*2*np.pi)*(Brob[:, :1] + Brob[:, 1:])

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'], Re=Re,
                   treat_nonl_explicit=True,
                   fvtd=fv_tmdp,
                   start_ssstokes=True,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput, prvoutpnts=400,
                   vfileprfx=proutdir+'vel_{0}_'.format(scheme),
                   pfileprfx=proutdir+'p_{0}_'.format(scheme))

    snu.solve_nse(**soldict)
    # print krylovdict['krpslvprms']['convstatsl']


if __name__ == '__main__':
    Re = 60
    Nts = 400
    tE = 1
    scaletest = 150
    testit(Re=Re, Nts=scaletest*Nts, tE=scaletest*tE,
           ParaviewOutput=True, scheme='TH', zerocontrol=False)
