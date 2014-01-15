#    This file is part of `dolfin_navier_scipy`.
#
#    `dolfin_navier_scipy` is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    `dolfin_navier_scipy` is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with `dolfin_navier_scipy`.  If not, see <http://www.gnu.org/licenses/>.

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
