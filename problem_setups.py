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

    Parameters
    ----------
    N : mesh parameter for the unitsquare (N gives 2*N*N triangles)
    vdgree : polynomial degree of the velocity basis functions, defaults to 2
    pdgree : polynomial degree of the pressure basis functions, defaults to 1

    Returns
    -------
    femp : a dictionary with the keys:
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

    # domains of observation and control
    odcoo = dict(xmin=0.45,
                 xmax=0.55,
                 ymin=0.5,
                 ymax=0.7)
    cdcoo = dict(xmin=0.4,
                 xmax=0.6,
                 ymin=0.2,
                 ymax=0.3)

    dfems.update(dict(cdcoo=cdcoo, odcoo=odcoo))

    return dfems


def cyl_fems(refinement_level=3, vdgree=2, pdgree=1):
    """
    dictionary for the fem items of the (unit) driven cavity

    Parameters
    ----------
    N : mesh parameter for the unitsquare (N gives 2*N*N triangles)
    vdgree : polynomial degree of the velocity basis functions,
        defaults to 2
    pdgree : polynomial degree of the pressure basis functions,
        defaults to 1

    Returns
    -------
    femp : a dictionary with the keys:
         * ``V``: FEM space of the velocity
         * ``Q``: FEM space of the pressure
         * ``diribcs``: list of the (Dirichlet) boundary conditions
         * ``dirip``: list of the (Dirichlet) boundary conditions
            for the pressure
         * ``fv``: right hand side of the momentum equation
         * ``fp``: right hand side of the continuity equation

    parts of the code were taken from the NSbench collection
    https://launchpad.net/nsbench

    __author__ = "Kristian Valen-Sendstad <kvs@simula.no>"
    __date__ = "2009-10-01"
    __copyright__ = "Copyright (C) 2009-2010 " + __author__
    __license__ = "GNU GPL version 3 or any later version"
    """

    # Constants related to the geometry
    bmarg = 1.e-3 + dolfin.DOLFIN_EPS
    xmin = 0.0
    xmax = 2.2
    ymin = 0.0
    ymax = 0.41
    xcenter = 0.2
    ycenter = 0.2
    radius = 0.05

    # Inflow boundary
    class InflowBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < xmin + bmarg

    # No-slip boundary
    class NoslipBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            dx = x[0] - xcenter
            dy = x[1] - ycenter
            r = dolfin.sqrt(dx*dx + dy*dy)
            return on_boundary and \
                (x[1] < ymin + bmarg or x[1] > ymax - bmarg or
                    r < radius + bmarg)

    # Outflow boundary
    class OutflowBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > xmax - bmarg

    ## meshes are available from https://launchpad.net/nsbench
    # Load mesh
    if refinement_level > 5:
        raise RuntimeError("No mesh available for refinement level {0}".
                           format(refinement_level))
    mesh = dolfin.Mesh("mesh/cylinder_%d.xml" % refinement_level)
    V = dolfin.VectorFunctionSpace(mesh, "CG", vdgree)
    Q = dolfin.FunctionSpace(mesh, "CG", pdgree)

    # Create right-hand side function
    fv = dolfin.Constant((0, 0))
    fp = dolfin.Constant(0)

    def initial_conditions(self, V, Q):
        u0 = dolfin.Constant((0, 0))
        p0 = dolfin.Constant(0)
        return u0, p0

    # Create inflow boundary condition
    g0 = dolfin.Expression(('4*(x[1]*(ymax-x[1]))/(ymax*ymax)', '0.0'),
                           ymax=ymax)
    bc0 = dolfin.DirichletBC(V, g0, InflowBoundary())

    # Create no-slip boundary condition
    g1 = dolfin.Constant((0, 0))
    bc1 = dolfin.DirichletBC(V, g1, NoslipBoundary())

    # Create outflow boundary condition for pressure
    g2 = dolfin.Constant(0)
    bc2 = dolfin.DirichletBC(Q, g2, OutflowBoundary())

    # Collect boundary conditions
    bcu = [bc0, bc1]
    bcp = [bc2]

    cylfems = dict(V=V,
                   Q=Q,
                   diribcs=bcu,
                   dirip=bcp,
                   fv=fv,
                   fp=fp)

    return cylfems
