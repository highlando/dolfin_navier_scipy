import dolfin
import numpy as np

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

dolfin.parameters.linear_algebra_backend = 'uBLAS'


def testit(problem='cylinderwake', N=2, nu=None, Re=1e2, Nts=1e3+1,
           ParaviewOutput=False, tE=1.0, scheme=None):

    nnewtsteps = 9  # n nwtn stps for vel comp
    vel_nwtn_tol = 1e-14
    tips = dict(t0=0.0, tE=tE, Nts=Nts)

    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
        = dnsps.get_sysmats(problem=problem, N=N, Re=Re,
                            bccontrol=True, nu=nu, scheme=scheme)
    proutdir = 'results/'
    ddir = 'data/'
    data_prfx = problem + '_N{0}_Re{1}_Nts{2}_tE{3}'.\
        format(N, femp['Re'], Nts, tE)

    palpha = 1e-5
    stokesmatsc['A'] = stokesmatsc['A'] + 1./palpha*stokesmatsc['Arob']
    Brob = 1./palpha*stokesmatsc['Brob']
    # import scipy.sparse as sps
    # # import matplotlib.pyplot as plt
    # # plt.figure(1)
    # # plt.spy(sps.csr_matrix(Brob))
    # # plt.figure(2)
    # # plt.spy(sps.csr_matrix(stokesmatsc['Arob']))
    # # plt.show()
    # print 'nnz of Arob:', stokesmatsc['Arob'].nnz
    # print 'nnz of Brob:', sps.csr_matrix(Brob).nnz
    # raise Warning('TODO: debug')

    def fv_tmdp(time=0, v=None, **kw):
        return np.sin(time)*(Brob[:, :1] - Brob[:, 1:]), None

    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs
    soldict.update(rhsd_vfrc)  # adding fvc, fpr
    soldict.update(tips)  # adding time integration params
    soldict.update(fv=rhsd_stbc['fv'], fp=rhsd_stbc['fp'],
                   N=N, nu=nu,
                   vel_nwtn_stps=nnewtsteps,
                   comp_nonl_semexp=True,
                   vel_nwtn_tol=vel_nwtn_tol,
                   fv_tmdp=fv_tmdp,
                   start_ssstokes=True,
                   get_datastring=None,
                   data_prfx=ddir+data_prfx,
                   paraviewoutput=ParaviewOutput,
                   vel_pcrd_stps=1,
                   clearprvdata=True,
                   vfileprfx=proutdir+'vel_{0}_'.format(scheme),
                   pfileprfx=proutdir+'p_{0}_'.format(scheme))

    snu.solve_nse(**soldict)
    # print krylovdict['krpslvprms']['convstatsl']


if __name__ == '__main__':
    # !!! bccontrol doesn't work for `scheme = 'CR'` !!!
    testit(problem='cylinderwake', N=2, Re=60, Nts=2e3, tE=4.,
           ParaviewOutput=True, scheme='TH')
