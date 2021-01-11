import os
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps


def testit(meshprfx='mesh/karman2D-outlets', meshlevel=1, proutdir='results/',
           problem='drivencavity', N=None, nu=1e-2, Re=None,
           t0=0.0, tE=1.0, Nts=1e2+1,
           ParaviewOutput=False, prvoutpnts=200,
           scheme='TH'):

    meshfile = meshprfx + '_lvl{0}.xml.gz'.format(meshlevel)
    physregs = meshprfx + '_lvl{0}_facet_region.xml.gz'.format(meshlevel)
    geodata = meshprfx + '_geo_cntrlbc.json'

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=False,
                          scheme=scheme, mergerhs=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))
    ddir = 'data/'
    data_prfx = problem + '{4}_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE, scheme)

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re

    tips = dict(t0=t0, tE=tE, Nts=Nts)

    try:
        os.chdir(ddir)
    except OSError:
        raise Warning('need "' + ddir + '" subdir for storing the data')
    os.chdir('..')

    plttrange = np.linspace(t0, tE, 101)
    plttrange = None

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
                   paraviewoutput=ParaviewOutput,
                   plttrange=plttrange, prvoutpnts=prvoutpnts,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    # v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)
    snu.solve_nse(**soldict)
    print('for plots check \nparaview ' + proutdir + 'vel___timestep.pvd')
    print('or \nparaview ' + proutdir + 'p___timestep.pvd')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshprefix", type=str,
                        help="prefix for the mesh files",
                        default='mesh/karman2D-outlets')
    parser.add_argument("--meshlevel", type=int,
                        help="mesh level", default=1)
    parser.add_argument("--Re", type=int,
                        help="Reynoldsnumber", default=100)
    parser.add_argument("--tE", type=float,
                        help="final time of the simulation", default=5.)
    parser.add_argument("--Nts", type=float,
                        help="number of time steps", default=8192)
    parser.add_argument("--scaletest", type=float,
                        help="scale the test size", default=1.)
    parser.add_argument("--paraviewframes", type=int,
                        help="number of outputs for paraview", default=200)
    args = parser.parse_args()
    print(args)
    scheme = 'TH'

    testit(problem='gen_bccont', Re=args.Re,
           meshprfx=args.meshprefix, meshlevel=args.meshlevel,
           t0=0., tE=args.scaletest*args.tE,
           Nts=np.int(args.scaletest*args.Nts),
           scheme=scheme, ParaviewOutput=True, prvoutpnts=args.paraviewframes)
