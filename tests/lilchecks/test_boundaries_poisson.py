import dolfin
import numpy as np
dolfin.parameters.linear_algebra_backend = "Eigen"

# Create classes for defining parts of the boundaries and the interior
# of the domain


class Left(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dolfin.near(x[0], 0.0)

# Initialize sub-domain instances
left = Left()

# Define mesh
mesh = dolfin.UnitSquareMesh(64, 64)

# Initialize mesh function for boundary domains
boundaries = dolfin.FacetFunction("size_t", mesh)
boundaries.set_all(0)
left.mark(boundaries, 1)

# Define function space and basis functions
V = dolfin.FunctionSpace(mesh, "CG", 2)
u = dolfin.TrialFunction(V)
v = dolfin.TestFunction(V)

# Define new measures associated with the interior domains and
# exterior boundaries
ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries)

aform = u*v*ds(1)
aform = dolfin.assemble(aform)

amatrob = dolfin.as_backend_type(aform).sparray()
amatrob.eliminate_zeros()

print('number of nonzeros in A: {0}'.format(amatrob.nnz))
print('norm of data of A: {0}'.format(np.linalg.norm(amatrob.data)))
