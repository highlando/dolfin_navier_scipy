import unittest
import dolfin
import scipy.sparse.linalg as spsla

import cont_obs_utils as cou


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


# unittests for the input and output operators


class TestInoutOpas(unittest.TestCase):

    # @unittest.skip("for now")
    def test_outopa_workingconfig(self):
        """ The innerproducts that assemble the output operator

        are accurately sampled for this parameter set (NV=25, NY=5)"""

        NV = 25
        NY = 5

        mesh = dolfin.UnitSquareMesh(NV, NV)
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

        exv = dolfin.Expression(('1', '1'))
        testv = dolfin.interpolate(exv, V)

        odcoo = dict(xmin=0.45,
                     xmax=0.55,
                     ymin=0.6,
                     ymax=0.8)

        # check the C
        MyC, My = cou.get_mout_opa(odcoo=odcoo, V=V, NY=NY, NV=NV)

        # signal space
        ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])

        Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

        y1 = dolfin.Function(Y)
        y2 = dolfin.Function(Y)

        testvi = testv.vector().array()
        testy = spsla.spsolve(My, MyC * testvi)

        y1 = dolfin.Expression('1')
        y1 = dolfin.interpolate(y1, Y)

        y2 = dolfin.Function(Y)
        y2.vector().set_local(testy[NY:])

        self.assertTrue(dolfin.errornorm(y2, y1) < 1e-14)

suite = unittest.TestLoader().loadTestsFromTestCase(TestInoutOpas)
unittest.TextTestRunner(verbosity=2).run(suite)
