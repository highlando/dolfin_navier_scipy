import dolfin
import numpy as np
import scipy.sparse as sps

try:
    dolfin.parameters.linear_algebra_backend = "Eigen"
except RuntimeError:
    dolfin.parameters.linear_algebra_backend = "uBLAS"

# Constants related to the geometry
bmarg = 1.e-3 + dolfin.DOLFIN_EPS
xmin = 0.0
xmax = 2.2
ymin = 0.0
ymax = 0.41
xcenter = 0.2
ycenter = 0.2
radius = 0.05

refinement_level = 3

verbose = True
verbose = False


class NoslipCylinderSurface(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        dx = x[0] - xcenter
        dy = x[1] - ycenter
        r = dolfin.sqrt(dx*dx + dy*dy)
        oncyl = on_boundary and r < radius + bmarg and dy < 0 and dx > 0
        if r < radius + bmarg:
            print r, oncyl, on_boundary
        return oncyl


class Left(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[0], 0.0)


def kernelfunc(element=None):
    class ContShapeTwo(dolfin.Expression):
        def eval(self, value, x):
            value[0], value[1] = 1., 1.
            if verbose:
                dx = x[0] - xcenter
                dy = x[1] - ycenter
                r = dolfin.sqrt(dx*dx + dy*dy)
                print x, dx, dy, r

        def value_shape(self):
            return (2,)
    return ContShapeTwo(element=element)

try:
    mesh = dolfin.Mesh("mesh/cylinder_{0}.xml.gz".format(refinement_level))
except RuntimeError:
    mesh = dolfin.Mesh("mesh/cylinder_{0}.xml".format(refinement_level))

# mesh = dolfin.UnitSquareMesh(64, 64)
V = dolfin.FunctionSpace(mesh, "CG", 2)
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

bcfun = kernelfunc(element=V.ufl_element())

Gamma = NoslipCylinderSurface()

boundaries = dolfin.FacetFunction("size_t", mesh)
boundaries.set_all(0)
Gamma.mark(boundaries, 1)

ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries)

# Robin boundary form
arob = u*v*ds(1)  # , subdomain_data=bparts)
# brob = dolfin.inner(v, bcfun) * dolfin.ds(0, subdomain_data=bparts)


def mat_dolfin2sparse(A):
    """get the csr matrix representing an assembled linear dolfin form

    """
    try:
        return dolfin.as_backend_type(A).sparray()
    except RuntimeError:  # `dolfin <= 1.5+` with `'uBLAS'` support
        rows, cols, values = A.data()
        return sps.csr_matrix((values, cols, rows))

amatrob = dolfin.assemble(arob)
amatrob = mat_dolfin2sparse(amatrob)
# bmatrob = dolfin.assemble(brob)

amatrob.eliminate_zeros()
# bmatrob = bmatrob.array().reshape((V.dim(), 1))

print 'number of nonzeros in A: {0}'.format(amatrob.nnz)
print 'norm of data of A: {0}'.format(np.linalg.norm(amatrob.data))
# print 'norm of b: {0}'.format(np.linalg.norm(bmatrob))
