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
import dolfin_navier_scipy.dolfin_to_sparrays as dts
import numpy as np


def get_sysmats(problem='drivencavity', N=10, scheme=None, ppin=None,
                Re=None, nu=None, ParaviewOutput=False):
    """ retrieve the system matrices for stokes flow

    Parameters
    ----------
    problem : {'drivencavity', 'cylinderwake'}
        problem class
    N : int
        mesh parameter
    nu : real, optional
        kinematic viscosity, is set to `L/Re` if `Re` is provided
    Re : real, optional
        Reynoldsnumber, is set to `L/nu` if `nu` is provided

    Returns
    -------
    femp : dict
        with the keys:
         * `V`: FEM space of the velocity
         * `Q`: FEM space of the pressure
         * `diribcs`: list of the (Dirichlet) boundary conditions
         * `bcinds`: indices of the boundary nodes
         * `bcvals`: values of the boundary nodes
         * `invinds`: indices of the inner nodes
         * `fv`: right hand side of the momentum equation
         * `fp`: right hand side of the continuity equation
         * `charlen`: characteristic length of the setup
         * `nu`: the kinematic viscosity
         * `Re`: the Reynolds number
         * `odcoo`: dictionary with the coordinates of the domain of \
                 observation
         * `cdcoo`: dictionary with the coordinates of the domain of \
         * `ppin` : {int, None}
                which dof of `p` is used to pin the pressure, typically \
                `-1` for internal flows, and `None` for flows with outflow
                         control
    stokesmatsc : dict
        a dictionary of the condensed matrices:
         * `M`: the mass matrix of the velocity space,
         * `A`: the stiffness matrix,
         * `JT`: the gradient matrix, and
         * `J`: the divergence matrix
         * `Jfull`: the uncondensed divergence matrix
    rhsd_vfrc : dict
        of the dirichlet and pressure fix reduced right hand sides
    rhsd_stbc : dict
        of the contributions of the boundary data to the rhs:
         * `fv`: contribution to momentum equation,
         * `fp`: contribution to continuity equation

    Examples
    --------
    femp, stokesmatsc, rhsd_vfrc, rhsd_stbc \
        = get_sysmats(problem='drivencavity', N=10, nu=1e-2)

    """

    problemdict = dict(drivencavity=drivcav_fems,
                       cylinderwake=cyl_fems)
    problemfem = problemdict[problem]
    femp = problemfem(N, scheme=scheme)

    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re
    else:
        Re = femp['charlen']/nu

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'], nu)

    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure if required
    if problem == 'cylinderwake':
        print 'cylinderwake: pressure need not be pinned'
        if ppin is not None:
            raise UserWarning('pinning the p will give wrong results')
    elif ppin is None:
        print 'pressure is not pinned - `J` may be singular for internal flow'
    elif ppin == -1:
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
        rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]
        print 'pressure pinned at last dof `-1`'
    else:
        raise NotImplementedError('Cannot pin `p` other than at `-1`')

    # reduce the matrices by resolving the BCs
    (stokesmatsc,
     rhsd_stbc,
     invinds,
     bcinds,
     bcvals) = dts.condense_sysmatsbybcs(stokesmats,
                                         femp['diribcs'])
    stokesmatsc.update({'Jfull': stokesmats['J']})

    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])

    # add the info on boundary and inner nodes
    bcdata = {'bcinds': bcinds,
              'bcvals': bcvals,
              'invinds': invinds,
              'ppin': ppin}
    femp.update(bcdata)
    femp.update({'nu': nu})
    femp.update({'Re': Re})

    return femp, stokesmatsc, rhsd_vfrc, rhsd_stbc


def drivcav_fems(N, vdgree=2, pdgree=1, scheme=None):
    """dictionary for the fem items of the (unit) driven cavity

    Parameters
    ----------
    N : int
        mesh parameter for the unitsquare (N gives 2*N*N triangles)
    vdgree : int, optional
        polynomial degree of the velocity basis functions, defaults to 2
    pdgree : int, optional
        polynomial degree of the pressure basis functions, defaults to 1
    scheme : {None, 'CR', 'TH'}
        the finite element scheme to be applied, 'CR' for Crouzieux-Raviart,\
        'TH' for Taylor-Hood, overrides `pdgree`, `vdgree`, defaults to `None`


    Returns
    -------
    femp : a dict
        of problem FEM description with the keys:
         * `V`: FEM space of the velocity
         * `Q`: FEM space of the pressure
         * `diribcs`: list of the (Dirichlet) boundary conditions
         * `fv`: right hand side of the momentum equation
         * `fp`: right hand side of the continuity equation
         * `charlen`: characteristic length of the setup
         * `odcoo`: dictionary with the coordinates of the domain of \
                 observation
         * `cdcoo`: dictionary with the coordinates of the domain of \
                 control
    """

    mesh = dolfin.UnitSquareMesh(N, N)
    if scheme == 'CR':
        print 'we use Crouzieux-Raviart elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    if scheme == 'TH':
        print 'we use Taylor-Hood elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
        Q = dolfin.FunctionSpace(mesh, "CG", 1)
    else:
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
                 fp=fp,
                 uspacedep=0,
                 charlen=1.0)

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


def cyl_fems(refinement_level=2, vdgree=2, pdgree=1, scheme=None):
    """
    dictionary for the fem items for the cylinder wake

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
         * `V`: FEM space of the velocity
         * `Q`: FEM space of the pressure
         * `diribcs`: list of the (Dirichlet) boundary conditions
         * `dirip`: list of the (Dirichlet) boundary conditions \
                 for the pressure
         * `fv`: right hand side of the momentum equation
         * `fp`: right hand side of the continuity equation
         * `charlen`: characteristic length of the setup
         * `odcoo`: dictionary with the coordinates of the \
                 domain of observatio
         * `cdcoo`: dictionary with the coordinates of the domain of control
         * `uspacedep`: int that specifies in what spatial direction Bu \
                changes. The remaining is constant

    Notes
    -----
    parts of the code were taken from the NSbench collection
    https://launchpad.net/nsbench

    |  __author__ = "Kristian Valen-Sendstad <kvs@simula.no>"
    |  __date__ = "2009-10-01"
    |  __copyright__ = "Copyright (C) 2009-2010 " + __author__
    |  __license__ = "GNU GPL version 3 or any later version"
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

    # boundary control at the cylinder
    # we define two symmetric wrt x-axis little outlets
    # via the radian of the center of the outlets and the extension
    centerrad = np.pi/4  # measured from the most downstream point (pi/2 = top)
    extensrad = np.pi/12  # radian measure of the extension of the outlets

    # bounding boxes for outlet domains
    if centerrad + extensrad/2 > np.pi/2 or centerrad - extensrad/2 < 0:
        raise NotImplementedError('1st outlet must lie in the 1st quadrant')
    b1xmin = xcenter + np.cos(centerrad + extensrad/2)
    b1ymax = ycenter + np.sin(centerrad + extensrad/2)
    b1xmax = xcenter + np.cos(centerrad - extensrad/2)
    b1ymin = ycenter + np.sin(centerrad - extensrad/2)
    # symmetry wrt x-axis
    b2xmin, b2xmax = b1xmin, b1xmax
    b2ymin, b2ymax = -b1ymax, -b1ymin

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

    # # meshes are available from https://launchpad.net/nsbench
    # Load mesh
    if refinement_level > 9:
        raise RuntimeError("No mesh available for refinement level {0}".
                           format(refinement_level))
    mesh = dolfin.Mesh("mesh/cylinder_%d.xml" % refinement_level)

    # scheme = 'CR'
    if scheme == 'CR':
        print 'we use Crouzieux-Raviart elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    else:
        V = dolfin.VectorFunctionSpace(mesh, "CG", vdgree)
        Q = dolfin.FunctionSpace(mesh, "CG", pdgree)

    # dolfin.plot(mesh)
    # dolfin.interactive(True)

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
                   fp=fp,
                   uspacedep=0,
                   charlen=0.1,
                   mesh=mesh)

    # domains of observation and control
    odcoo = dict(xmin=0.6,
                 xmax=0.7,
                 ymin=0.15,
                 ymax=0.25)
    cdcoo = dict(xmin=0.27,
                 xmax=0.32,
                 ymin=0.15,
                 ymax=0.25)

    cylfems.update(dict(cdcoo=cdcoo, odcoo=odcoo))

    return cylfems
