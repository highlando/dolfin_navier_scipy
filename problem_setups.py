# This file is part of `dolfin_navier_scipy`.
#
# `dolfin_navier_scipy` is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# `dolfin_navier_scipy` is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with `dolfin_navier_scipy`.
# If not, see <http://www.gnu.org/licenses/>.

import dolfin


def drivcav_fems(N, vdgree=2, pdgree=1):
    """dictionary for the fem items of the (unit) driven cavity

    :param N:
        mesh parameter for the unitsquare (N gives 2*N*N triangles)
    :param vdgree:
        polynomial degree of the velocity basis functions, defaults to 2
    :param pdgree:
        polynomial degree of the pressure basis functions, defaults to 1

    :return:
        a dictionary with the keys:
        * ``V``: FEM space of the velocity
        * ``Q``: FEM space of the pressure
        * ``diribcs``: list of the (Dirichlet) boundary conditions
        * ``fv``: right hand side of the momentum equation
        * ``fp``: right hand side of the continuity equation
    """

    mesh = dolfin.UnitSquareMesh(N, N)
    V = dolfin.VectorFunctionSpace(mesh, "CG", vdgree)
    Q = dolfin.FunctionSpace(mesh, "CG", pdgree)

    # Boundaries
    def top(x, on_boundary):
        return x[1] > 1.0 - dolfin.DOLFIN_EPS

    def leftbotright(x, on_boundary):
        return (x[0] > 1.0 - dolfin.DOLFIN_EPS
                or x[1] < dolfin.DOLFIN_EPS
                or x[0] < dolfin.DOLFIN_EPS)

    # No-slip boundary condition for velocity
    noslip = dolfin.Constant((0.0, 0.0))
    bc0 = dolfin.DirichletBC(V, noslip, leftbotright)
    # Boundary condition for velocity at the lid
    lid = dolfin.Constant(("1", "0.0"))
    bc1 = dolfin.DirichletBC(V, lid, top)
    # Collect boundary conditions
    diribcs = [bc0, bc1]
    # rhs of momentum eqn
    fv = dolfin.Constant((0.0, 0.0))
    # rhs of the continuity eqn
    fp = dolfin.Constant(0.0)

    dfems = dict(V=V,
                 Q=Q,
                 diribcs=diribcs,
                 fv=fv,
                 fp=fp)

    return dfems

__author__ = "Kristian Valen-Sendstad <kvs@simula.no>"
__date__ = "2009-10-01"
__copyright__ = "Copyright (C) 2009-2010 " + __author__
__license__ = "GNU GPL version 3 or any later version"

# Modified by Anders Logg, 2010.

from problembase import *
from numpy import array

# Constants related to the geometry
bmarg = 1.e-3 + DOLFIN_EPS
xmin = 0.0
xmax = 2.2
ymin = 0.0
ymax = 0.41
xcenter = 0.2
ycenter = 0.2
radius = 0.05

# Inflow boundary
class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < xmin + bmarg

# No-slip boundary
class NoslipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        dx = x[0] - xcenter
        dy = x[1] - ycenter
        r = sqrt(dx*dx + dy*dy)
        return on_boundary and \
               (x[1] < ymin + bmarg or x[1] > ymax - bmarg or \
                r < radius + bmarg)

# Outflow boundary
class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] > xmax - bmarg

# Problem definition
class Problem(ProblemBase):

    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # Load mesh
        refinement_level = options["refinement_level"]
        if refinement_level > 5:
            raise RuntimeError, "No mesh available for refinement level %d" % refinement_level

        self.mesh = Mesh("data/cylinder_%d.xml.gz" % refinement_level)


        # Create right-hand side function
        self.f =  Constant((0, 0))

        # Set viscosity (Re = 1000)
        self.nu = 1.0 / 1000.0

        # Characteristic velocity in the domain (used to determinde timestep)
        self.U = 3.5

        # Set end time
        self.T  = 8.0

    def initial_conditions(self, V, Q):

        u0 = Constant((0, 0))
        p0 = Constant(0)

        return u0, p0

    def boundary_conditions(self, V, Q, t):

        # Create inflow boundary condition
        self.g0 = Expression(('4*Um*(x[1]*(ymax-x[1]))*sin(pi*t/8.0)/(ymax*ymax)', '0.0'),
                             Um=1.5, ymax=ymax, t=t)
        self.b0 = InflowBoundary()
        bc0 = DirichletBC(V, self.g0, self.b0)

        # Create no-slip boundary condition
        self.b1 = NoslipBoundary()
        self.g1 = Constant((0, 0))
        bc1     = DirichletBC(V, self.g1, self.b1)

        # Create outflow boundary condition for pressure
        self.b2 = OutflowBoundary()
        self.g2 = Constant(0)
        bc2     = DirichletBC(Q, self.g2, self.b2)

        # Collect boundary conditions
        bcu = [bc0, bc1]
        bcp = [bc2]

        return bcu, bcp

    def update(self, t, u, p):
        self.g0.t = t

    def functional(self, t, u, p):

        if t < self.T:
            return 0.0

        x1 = array((xcenter - radius - DOLFIN_EPS, ycenter))
        x2 = array((xcenter + radius + DOLFIN_EPS, ycenter))

        return p(x1) - p(x2)

    def reference(self, t):
        
        if t < self.T:
            return 0.0
        
        return -0.111444953719

    def __str__(self):
        return "Cylinder"
