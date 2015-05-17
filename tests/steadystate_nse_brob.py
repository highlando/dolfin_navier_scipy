import dolfin
import os

import dolfin_navier_scipy.dolfin_to_sparrays as dts
import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def testit(problem='drivencavity', N=None, nu=1e-2, Re=1e2,
           nnwtnstps=9, npcrdstps=5):

    palpha = 1e-8

    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
        = dnsps.get_sysmats(problem=problem, N=N, Re=Re,
                            bccontrol=True, scheme='TH')

    nu = femp['charlen']/Re

    stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
    b_mat = 1./palpha*stokesmatsc['Brob']

    vel_nwtn_tol = 1e-14
    # prefix for data files
    data_prfx = problem
    # dir to store data
    ddir = 'data/'
    # paraview output
    ParaviewOutput = True
    proutdir = 'results/'

    if ParaviewOutput:
        curwd = os.getcwd()
        try:
            os.chdir(proutdir)
            # for fname in glob.glob(data_prfx + '*'):
            #     os.remove(fname)
            os.chdir(curwd)
        except OSError:
            raise Warning('the ' + proutdir + ' subdir for storing the' +
                          ' output does not exist. Make it yourself' +
                          ' or set paraviewoutput=False')

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(fv=rhsd_stbc['fv'], fp=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnwtnstps,
                   vel_pcrd_stps=npcrdstps,
                   vel_nwtn_tol=vel_nwtn_tol,
                   ddir=ddir, get_datastring=None,
                   clearprvdata=True,
                   data_prfx=data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vfileprfx=proutdir+'vel_',
                   pfileprfx=proutdir+'p_')

#
# compute the uncontrolled steady state Navier-Stokes solution
#
    v_ss_nse, list_norm_nwtnupd = snu.solve_steadystate_nse(**soldict)


if __name__ == '__main__':
    # testit(N=25, nu=3e-4)
    # testit(problem='cylinderwake', N=3, nu=2e-3)
    # testit(problem='drivencavity', N=25, Re=500)
    testit(problem='cylinderwake', N=4, Re=1.8e2,
           nnwtnstps=5, npcrdstps=15)
