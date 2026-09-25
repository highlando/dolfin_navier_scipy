import os
import logging
# import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

meshprfx = 'mesh/karman2D-outlets'
meshlevel = 3
meshfile = meshprfx + '_lvl{0}.xml.gz'.format(meshlevel)
physregs = meshprfx + '_lvl{0}_facet_region.xml.gz'.format(meshlevel)
geodata = meshprfx + '_geo_cntrlbc.json'
proutdir = 'results/'


def testit(problem='drivencavity', N=None, nu=1e-2, Re=None,
           t0=0.0, tE=1.0, Nts=1e2+1, ParaviewOutput=False, scheme='TH',
           symmetricgrad=True):

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=False,
                          scheme=scheme, mergerhs=True,
                          gradvsymmtrc=symmetricgrad,
                          meshparams=dict(strtomeshfile=meshfile,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))
    ddir = 'data/'
    data_prfx = problem + '{4}_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE, scheme)

    print('computing `c_mat`...')
    # import sadptprj_riclyap_adi.lin_alg_utils as lau
    import distributed_control_fenics.cont_obs_utils as cou
    c_mat = cou.get_mout_opa(odcoo=femp['odcoo'], V=femp['V'], mfgrid=(3, 1))
    # c_mat = lau.apply_massinv(y_masmat, mc_mat, output='sparse')
    # restrict the operator to the inner nodes
    c_mat = c_mat[:, femp['invinds']][:, :]

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    svgstr = '' if symmetricgrad else '_nsmg'
    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=N, nu=nu,
                   start_ssstokes=True,
                   get_datastring=None,
                   treat_nonl_explicit=True,
                   dbcinds=femp['dbcinds'], dbcvals=femp['dbcvals'],
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput, prvoutpnts=100,
                   vfileprfx=proutdir+f'vel{svgstr}msh{meshlevel}_',
                   pfileprfx=proutdir+f'p{svgstr}msh{meshlevel}_')

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    # v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)
    snu.solve_nse(**soldict)
    print('for plots check \nparaview ' + proutdir +
          f'vel{svgstr}msh{meshlevel}___timestep.pvd')
    print('or \nparaview ' + proutdir +
          f'p{svgstr}msh{meshlevel}___timestep.pvd')


if __name__ == '__main__':
    scheme = 'TH'
    Re = 30
    t0, tE, Nts = 0., 5., 4*2048
    scaletest = 2.
    symgrad = False
    symgrad = True

    testit(problem='gen_bccont', Re=Re,
           t0=scaletest*t0, tE=scaletest*tE, Nts=int(scaletest*Nts),
           scheme=scheme, ParaviewOutput=True, symmetricgrad=symgrad)
