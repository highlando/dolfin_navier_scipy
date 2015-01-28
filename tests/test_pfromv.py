import unittest
import numpy as np

import dolfin_navier_scipy.problem_setups as dnsps
import dolfin_navier_scipy.stokes_navier_utils as snu


class StokNavUtsFunctions(unittest.TestCase):

    def setUp(self):
        self.N = 2
        self.Re = 50
        self.scheme = 'TH'
        self.ppin = None

    def test_pfv(self):
        """check the computation of p from a given v

        """

        femp, stokesmatsc, rhsd_vfrc, \
            rhsd_stbc, data_prfx, ddir, proutdir \
            = dnsps.get_sysmats(problem='cylinderwake', N=self.N,
                                Re=self.Re, scheme=self.scheme)

        Mc, Ac = stokesmatsc['M'], stokesmatsc['A']
        BTc, Bc = stokesmatsc['JT'], stokesmatsc['J']
        print Bc.shape

        invinds = femp['invinds']

        fv, fp = rhsd_stbc['fv'], rhsd_stbc['fp']
        print np.linalg.norm(fv), np.linalg.norm(fp)
        inivdict = dict(A=Ac, J=Bc, JT=BTc, M=Mc, ppin=self.ppin, fv=fv, fp=fp,
                        return_vp=True, V=femp['V'], clearprvdata=True,
                        invinds=invinds, diribcs=femp['diribcs'])
        vp_init = snu.solve_steadystate_nse(**inivdict)[0]

        NV = Bc.shape[1]

        pfv = snu.get_pfromv(v=vp_init[:NV, :], V=femp['V'],
                             M=Mc, A=Ac, J=Bc, fv=fv,
                             invinds=femp['invinds'], diribcs=femp['diribcs'])
        self.assertTrue(np.allclose(pfv, vp_init[NV:, :]))
