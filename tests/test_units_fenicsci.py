import unittest
import sympy as smp
import numpy as np
import dolfin

# unittests for the suite
# if not specified otherwise we use the unit square
# with 0-Dirichlet BCs with a known solution


class OptConPyFunctions(unittest.TestCase):

    def setUp(self):

        self.mesh = dolfin.UnitSquareMesh(24, 24)
        self.V = dolfin.VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = dolfin.FunctionSpace(self.mesh, "CG", 1)
        self.nu = 1e-5

        x, y, t, nu, om = smp.symbols('x,y,t,nu,om')
        ft = smp.sin(om * t)
        u_x = ft * x * x * (1 - x) * (1 - x) * 2 * y * (1 - y) * (2 * y - 1)
        u_y = ft * y * y * (1 - y) * (1 - y) * 2 * x * (1 - x) * (1 - 2 * x)
        p = ft * x * (1 - x) * y * (1 - y)

        # div u --- should be zero!!
        self.assertEqual(smp.simplify(smp.diff(u_x, x) + smp.diff(u_y, y)), 0)

        self.u_x = u_x
        self.u_y = u_y
        self.p = p

        def sympy2expression(term):
            '''Translate a SymPy expression to a FEniCS expression string.
               '''
            # This is somewhat ugly:
            # First replace the variables r, z, by something
            # that probably doesn't appear anywhere else,
            # e.g., RRR, ZZZ, then
            # convert this into a string,
            # and then replace the substrings RRR, ZZZ
            # by x[0], x[1], respectively.
            exp = smp.printing.ccode(term.subs('x', 'XXX').subs('y', 'YYY')) \
                .replace('M_PI', 'pi') \
                .replace('XXX', 'x[0]').replace('YYY', 'x[1]')
            return exp

        # dotu_x = smp.simplify(smp.diff(u_x, t))
        # dotu_y = smp.simplify(smp.diff(u_y, t))

        # diffu_x = smp.simplify(
        #     nu * (smp.diff(u_x, x, x) + smp.diff(u_x, y, y)))
        # diffu_y = smp.simplify(
        #     nu * (smp.diff(u_y, x, x) + smp.diff(u_y, y, y)))

        # dp_x = smp.simplify(smp.diff(p, x))
        # dp_y = smp.simplify(smp.diff(p, y))

        # adv_x = smp.simplify(u_x * smp.diff(u_x, x) + u_y * smp.diff(u_x, y))
        # adv_y = smp.simplify(u_x * smp.diff(u_y, x) + u_y * smp.diff(u_y, y))

        self.F = dolfin.Expression(('0', '0'), element=self.V.ufl_element())
        a = sympy2expression(u_x)
        b = sympy2expression(u_y)

        self.fenics_sol_u = dolfin.Expression((a, b), t=0.0, om=1.0,
                                              element=self.V.ufl_element())

    def test_linearized_mat_NSE_form(self):
        """check the conversion: dolfin form <-> numpy arrays

          and the linearizations"""

        import dolfin_navier_scipy.dolfin_to_sparrays as dts

        u = self.fenics_sol_u
        u.t = 1.0
        ufun = dolfin.project(u, self.V)
        uvec = ufun.vector().get_local().reshape(len(ufun.vector()), 1)

        N1, N2, fv = dts.get_convmats(u0_dolfun=ufun, V=self.V)
        conv = dts.get_convvec(u0_dolfun=ufun, V=self.V)

        self.assertTrue(np.allclose(conv, N1 * uvec))
        self.assertTrue(np.allclose(conv, N2 * uvec))

    def test_expand_condense_vfuncs(self):
        """check the expansion of vectors to dolfin funcs

        """
        from dolfin_navier_scipy.dolfin_to_sparrays import expand_vp_dolfunc

        u = dolfin.Expression(('x[1]', '0'), element=self.V.ufl_element())
        ufun = dolfin.project(u, self.V, solver_type='lu')
        uvec = ufun.vector().get_local().reshape(len(ufun.vector()), 1)

        # Boundaries
        def top(x, on_boundary):
            return x[1] > 1.0 - dolfin.DOLFIN_EPS

        def leftbotright(x, on_boundary):
            return (x[0] > 1.0 - dolfin.DOLFIN_EPS
                    or x[1] < dolfin.DOLFIN_EPS
                    or x[0] < dolfin.DOLFIN_EPS)

        # No-slip boundary condition for velocity
        noslip = u
        bc0 = dolfin.DirichletBC(self.V, noslip, leftbotright)
        # Boundary condition for velocity at the lid
        lid = u
        bc1 = dolfin.DirichletBC(self.V, lid, top)
        # Collect boundary conditions
        diribcs = [bc0, bc1]
        bcinds = []
        for bc in diribcs:
            bcdict = bc.get_boundary_values()
            bcinds.extend(bcdict.keys())

        # indices of the innernodes
        innerinds = np.setdiff1d(range(self.V.dim()),
                                 bcinds).astype(np.int32)

        # take only the inner nodes
        uvec_condensed = uvec[innerinds, ]

        v, p = expand_vp_dolfunc(V=self.V, vc=uvec_condensed,
                                 invinds=innerinds, diribcs=diribcs)

        vvec = v.vector().get_local().reshape(len(v.vector()), 1)

        self.assertTrue(np.allclose(uvec, vvec))

    def test_conv_asquad(self):
        import dolfin_navier_scipy as dns
        from dolfin import dx, grad, inner
        import dolfin_navier_scipy.dolfin_to_sparrays as dts

        femp, stokesmatsc, rhsd = \
            dns.problem_setups.get_sysmats(problem='drivencavity', nu=1e-2,
                                           mergerhs=True, meshparams={'N': 15})

        invinds = femp['invinds']
        V = femp['V']

        hmat = dns.dolfin_to_sparrays.\
            ass_convmat_asmatquad(W=femp['V'], invindsw=invinds)

        xexp = '(1-x[0])*x[0]*(1-x[1])*x[1]*x[0]+2'
        yexp = '(1-x[0])*x[0]*(1-x[1])*x[1]*x[1]+1'
        # yexp = 'x[0]*x[0]*x[1]*x[1]'

        f = dolfin.Expression((xexp, yexp), element=self.V.ufl_element())

        u = dolfin.interpolate(f, V)
        uvec = np.atleast_2d(u.vector().get_local()).T

        uvec_gamma = uvec.copy()
        uvec_gamma[invinds] = 0
        u_gamma = dolfin.Function(V)
        u_gamma.vector().set_local(uvec_gamma)

        uvec_i = 0 * uvec
        uvec_i[invinds, :] = uvec[invinds]
        u_i = dolfin.Function(V)
        u_i.vector().set_local(uvec_i)

        # Assemble the 'actual' form
        w = dolfin.TrialFunction(V)
        wt = dolfin.TestFunction(V)
        nform = dolfin.assemble(inner(grad(w) * u, wt) * dx)
        # rows, cols, values = nform.data()
        nmat = dts.mat_dolfin2sparse(nform)
        # consider only the 'inner' equations
        nmatrc = nmat[invinds, :][:, :]

        # the boundary terms
        N1, N2, fv = dns.dolfin_to_sparrays.\
            get_convmats(u0_dolfun=u_gamma, V=V)

        # print np.linalg.norm(nmatrc * uvec_i + nmatrc * uvec_gamma)
        classicalconv = nmatrc * uvec
        quadconv = (hmat * np.kron(uvec[invinds], uvec[invinds])
                    + ((N1 + N2) * uvec_i)[invinds, :] + fv[invinds, :])
        self.assertTrue(np.allclose(classicalconv, quadconv))
        # print 'consistency tests'
        self.assertTrue((np.linalg.norm(uvec[invinds])
                         - np.linalg.norm(uvec_i)) < 1e-14)
        self.assertTrue(np.linalg.norm(uvec - uvec_gamma - uvec_i) < 1e-14)


if __name__ == '__main__':
    unittest.main()
