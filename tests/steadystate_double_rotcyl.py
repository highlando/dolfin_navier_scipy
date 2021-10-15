# import numpy as np

import dolfin
import matplotlib.pyplot as plt

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.problem_setups as dnsps

geodata = 'mesh/2D-double-rotcyl_geo_cntrlbc.json'
proutdir = 'results/'


def testit(problem=None, nu=None, charvel=None, Re=None,
           meshlvl=1, gradvsymmtrc=True,
           rho=1.,
           ParaviewOutput=False, scheme='TH'):

    meshfile = 'mesh/2D-double-rotcyl_lvl{0}.xml.gz'.format(meshlvl)
    physregs = 'mesh/2D-double-rotcyl_lvl{0}_facet_region.xml.gz'.\
        format(meshlvl)
    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', nu=nu, Re=Re,
                          charvel=charvel, gradvsymmtrc=gradvsymmtrc,
                          scheme=scheme, mergerhs=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          movingwallcntrl=False,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))

    ddir = 'data/'
    data_prfx = problem+'{2}_mesh{0}_Re{1}'.format(meshlvl, femp['Re'], scheme)

    NP, NV = stokesmatsc['J'].shape
    print('NV + NP : {0} + {1} = {2}'.format(NV, NP, NV+NP))

    soldict = stokesmatsc  # containing A, J, JT
    # soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(invinds=femp['invinds'], V=femp['V'], Q=femp['Q'])
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=meshlvl, nu=nu,
                   vel_nwtn_tol=5e-13,
                   vel_pcrd_stps=30,
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

    plt.figure(1)
    dolfin.plot(vss)
    plt.figure(2)
    dolfin.plot(dynpss)
    plt.show()


if __name__ == '__main__':
    meshlvl = 2
    nu = 1e-3
    rho = 1.
    charvel = 1.
    Re = 40

    # scheme = 'CR'
    # testit(problem='gen_bccont', nu=nu, charvel=charvel,
    #        rho=rho, meshlvl=meshlvl, gradvsymmtrc=False,
    #        scheme=scheme, ParaviewOutput=True)

    scheme = 'TH'
    # testit(problem='2D-double-cyl', nu=nu, charvel=charvel,
    #        rho=rho, meshlvl=meshlvl,  # gradvsymmtrc=False,
    #        scheme=scheme, ParaviewOutput=True)

    testit(problem='2D-double-cyl', Re=Re, charvel=charvel,
           rho=rho, meshlvl=meshlvl,  # gradvsymmtrc=False,
           scheme=scheme, ParaviewOutput=True)
