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
import json

__all__ = ['get_sysmats',
           'drivcav_fems',
           'cyl_fems',
           'gen_bccont_fems',
           'cyl3D_fems']


def get_sysmats(problem='gen_bccont', scheme=None, ppin=None,
                Re=None, nu=None, charvel=1., gradvsymmtrc=True,
                bccontrol=False, mergerhs=False,
                onlymesh=False, meshparams={}):
    """ retrieve the system matrices for stokes flow

    Parameters
    ----------
    problem : {'drivencavity', 'cylinderwake', 'gen_bccont', 'cylinder_rot'}
        problem class
    N : int
        mesh parameter
    nu : real, optional
        kinematic viscosity, is set to `L/Re` if `Re` is provided
    Re : real, optional
        Reynoldsnumber, is set to `L/nu` if `nu` is provided
    bccontrol : boolean, optional
        whether to consider boundary control via penalized Robin \
        defaults to `False`
    mergerhs : boolean, optional
        whether to merge the actual rhs and the contribution from the \
        boundary conditions into one rhs, defaults to `False`
    onlymesh : boolean, optional
        whether to only return `femp`, containing the mesh and FEM spaces, \
        defaults to `False`

    Returns
    -------
    femp : dict
        with the keys:
         * `V`: FEM space of the velocity
         * `Q`: FEM space of the pressure
         * `diribcs`: list of the (Dirichlet) boundary conditions
         * `dbcinds`: indices of the boundary nodes
         * `dbcvals`: values of the boundary nodes
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
         * `MP`: the mass matrix of the pressure space,
         * `A`: the stiffness matrix,
         * `JT`: the gradient matrix, and
         * `J`: the divergence matrix
         * `Jfull`: the uncondensed divergence matrix
        and, if `bccontrol=True`, the boundary control matrices that weakly \
        impose `Arob*v = Brob*u`, where
         * `Arob`: contribution to `A`
         * `Brob`: input operator
    `if mergerhs`
    rhsd : dict
        `rhsd_vfrc` and `rhsd_stbc` merged
    `else`
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
                       cylinderwake=cyl_fems,
                       cylinderwake3D=cyl3D_fems,
                       gen_bccont=gen_bccont_fems)

    if problem == 'cylinderwake' or problem == 'gen_bccont':
        meshparams.update(dict(inflowvel=charvel))

    if problem == 'cylinder_rot':
        problemfem = gen_bccont_fems
        meshparams.update(dict(movingwallcntrl=True))
        meshparams.update(dict(inflowvel=charvel))
    else:
        problemfem = problemdict[problem]

    femp = problemfem(scheme=scheme, bccontrol=bccontrol, **meshparams)

    if onlymesh:
        return femp

    # setting some parameters
    if Re is not None:
        nu = charvel*femp['charlen']/Re
    else:
        Re = charvel*femp['charlen']/nu

    if bccontrol:
        try:
            cbshapefuns = femp['contrbcsshapefuns']
            cbclist = femp['contrbcssubdomains']
            cbds = None
        except KeyError:
            cbshapefuns = femp['contrbcsshapefuns']
            cbclist = None
            cbds = femp['cntrbcsds']
    else:
        cbclist, cbshapefuns, cbds = None, None, None

    try:
        outflowds = femp['outflowds']
    except KeyError:
        outflowds = None

    stokesmats = dts.get_stokessysmats(femp['V'], femp['Q'], nu,
                                       cbclist=cbclist, cbds=cbds,
                                       gradvsymmtrc=gradvsymmtrc,
                                       outflowds=outflowds,
                                       cbshapefuns=cbshapefuns,
                                       bccontrol=bccontrol)

    rhsd_vf = dts.setget_rhs(femp['V'], femp['Q'],
                             femp['fv'], femp['fp'], t=0)

    # remove the freedom in the pressure if required
    if problem == 'cylinderwake':
        print('cylinderwake: pressure need not be pinned')
        if ppin is not None:
            raise UserWarning('pinning the p will give wrong results')
    elif ppin is None:
        print('pressure is not pinned - `J` may be singular for internal flow')
    elif ppin == -1:
        stokesmats['J'] = stokesmats['J'][:-1, :][:, :]
        stokesmats['JT'] = stokesmats['JT'][:, :-1][:, :]
        rhsd_vf['fp'] = rhsd_vf['fp'][:-1, :]
        print('pressure pinned at last dof `-1`')
    else:
        raise NotImplementedError('Cannot pin `p` other than at `-1`')

    # reduce the matrices by resolving the BCs
    try:
        (stokesmatsc, rhsd_stbc, invinds, _,
         _) = dts.condense_sysmatsbybcs(stokesmats, femp['diribcs'])
    except KeyError:  # can also just provide indices and values
        (stokesmatsc, rhsd_stbc, invinds, _, _) = \
            dts.condense_sysmatsbybcs(stokesmats, dbcinds=femp['dbcinds'],
                                      dbcvals=femp['dbcvals'])
    stokesmatsc.update({'Jfull': stokesmats['J']})

    # pressure freedom and dirichlet reduced rhs
    rhsd_vfrc = dict(fpr=rhsd_vf['fp'], fvc=rhsd_vf['fv'][invinds, ])
    if bccontrol:
        Arob, fvrob = dts.condense_velmatsbybcs(stokesmats['amatrob'],
                                                dbcinds=femp['dbcinds'],
                                                dbcvals=femp['dbcvals'])
        if np.linalg.norm(fvrob) > 1e-15:
            raise UserWarning('diri and control bc must not intersect')

        Brob = stokesmats['bmatrob'][invinds, :]
        stokesmatsc.update({'Brob': Brob, 'Arob': Arob})

    # add the info on boundary and inner nodes
    bcdata = {'invinds': invinds,
              'ppin': ppin}
    femp.update(bcdata)
    femp.update({'nu': nu})
    femp.update({'Re': Re})

    if mergerhs:
        rhsd = dict(fv=rhsd_vfrc['fvc']+rhsd_stbc['fv'],
                    fp=rhsd_vfrc['fpr']+rhsd_stbc['fp'])
        return femp, stokesmatsc, rhsd
    else:
        return femp, stokesmatsc, rhsd_vfrc, rhsd_stbc


def drivcav_fems(N, vdgree=2, pdgree=1, scheme=None, bccontrol=None):
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
    bccontrol : boolean, optional
        whether to consider boundary control via penalized Robin \
        defaults to false. TODO: not implemented yet but we need it here \
        for consistency

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
        # print 'we use Crouzieux-Raviart elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    if scheme == 'TH':
        # print 'we use Taylor-Hood elements !'
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


def cyl_fems(refinement_level=2, vdgree=2, pdgree=1, scheme=None,
             inflowvel=1., bccontrol=False, verbose=False):
    """
    dictionary for the fem items for the cylinder wake

    Parameters
    ----------
    vdgree : polynomial degree of the velocity basis functions,
        defaults to 2
    pdgree : polynomial degree of the pressure basis functions,
        defaults to 1
    scheme : {None, 'CR', 'TH'}
        the finite element scheme to be applied, 'CR' for Crouzieux-Raviart,\
        'TH' for Taylor-Hood, overrides `pdgree`, `vdgree`, defaults to `None`
    bccontrol : boolean, optional
        whether to consider boundary control via penalized Robin \
        defaults to `False`

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
                 domain of observation
         * `cdcoo`: dictionary with the coordinates of the domain of control
         * `uspacedep`: int that specifies in what spatial direction Bu \
                changes. The remaining is constant
         * `bcsubdoms`: list of subdomains that define the segments where \
                 the boundary control is applied

    Notes
    -----
    TODO: `inflowvel` as input is there for consistency but not processed

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
    centerrad = np.pi/3  # measured from the most downstream point (pi/2 = top)
    extensrad = np.pi/6  # radian measure of the extension of the outlets

    # bounding boxes for outlet domains
    if centerrad + extensrad/2 > np.pi/2 or centerrad - extensrad/2 < 0:
        raise NotImplementedError('1st outlet must lie in the 1st quadrant')
    b1xmin = xcenter + radius*np.cos(centerrad + extensrad/2)
    b1ymax = ycenter + radius*np.sin(centerrad + extensrad/2)
    b1xmax = xcenter + radius*np.cos(centerrad - extensrad/2)
    b1ymin = ycenter + radius*np.sin(centerrad - extensrad/2)
    # symmetry wrt x-axis
    b2xmin, b2xmax = b1xmin, b1xmax
    b2ymin = ycenter - radius*np.sin(centerrad + extensrad/2)
    b2ymax = ycenter - radius*np.sin(centerrad - extensrad/2)

    # vectors from the center to the control domain corners
    # we need them to define/parametrize the control shape functions
    b1base = np.array([[b1xmax - xcenter], [b1ymin - ycenter]])
    b2base = np.array([[b2xmin - xcenter], [b2ymin - ycenter]])

    # normal vectors of the control domain (considered as a straight line)
    centvec = np.array([[xcenter], [ycenter]])
    b1tang = np.array([[b1xmax - b1xmin], [b1ymin - b1ymax]])
    b2tang = np.array([[b2xmin - b2xmax], [b2ymin - b2ymax]])

    rotby90 = np.array([[0, -1.], [1., 0]])
    b1normal = rotby90.dot(b1tang) / np.linalg.norm(b1tang)
    b2normal = rotby90.dot(b2tang) / np.linalg.norm(b2tang)

    if verbose:
        print('centvec :', centvec.flatten(), ' b1base', b1base.flatten())
        print(b1xmin, b1xmax, b1ymin, b1ymax)
        print(b2xmin, b2xmax, b2ymin, b2ymax)
        print(b1base, np.linalg.norm(b1base))
        print(b1tang)
        print(b1normal)
        print(b2base, np.linalg.norm(b2base))
        print(b2tang)
        print(b2normal)
        print('diameter of the outlet', radius*2*np.sin(extensrad/2))
        print('midpoint of the outlet 1 secant: [{0}, {1}]'.
              format(centvec[0]+radius*np.cos(centerrad),
                     centvec[1]+radius*np.sin(centerrad)))
        print('midpoint of the outlet 2 secant: [{0}, {1}]'.
              format(centvec[0]+radius*np.cos(centerrad),
                     centvec[1]-radius*np.sin(centerrad)))
        print('angle of midpoint vec 1 and x-axis', np.rad2deg(centerrad))

    def insidebbox(x, whichbox=None):
        inbbone = (x[0] > b1xmin and x[0] < b1xmax
                   and x[1] > b1ymin and x[1] < b1ymax)
        inbbtwo = (x[0] > b2xmin and x[0] < b2xmax
                   and x[1] > b2ymin and x[1] < b2ymax)
        if whichbox is None:
            return inbbone or inbbtwo
        if whichbox == 1:
            return inbbone
        if whichbox == 2:
            return inbbtwo

    # Inflow boundary
    class InflowBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] < xmin + bmarg

    # No-slip boundary
    class NoslipChannelWalls(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[1] < ymin + bmarg or x[1] > ymax - bmarg)

    class NoslipCylinderSurface(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            dx = x[0] - xcenter
            dy = x[1] - ycenter
            r = dolfin.sqrt(dx*dx + dy*dy)
            if bccontrol:
                notinbbx = not insidebbox(x)
                return on_boundary and r < radius + bmarg and notinbbx
            else:
                return on_boundary and r < radius + bmarg

    # Outflow boundary
    class OutflowBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > xmax - bmarg

    # # meshes are available from https://launchpad.net/nsbench
    # Load mesh
    if refinement_level > 9:
        raise RuntimeError("No mesh available for refinement level {0}".
                           format(refinement_level))
    try:
        mesh = dolfin.Mesh("mesh/cylinder_%d.xml.gz" % refinement_level)
    except RuntimeError:
        mesh = dolfin.Mesh("mesh/cylinder_%d.xml" % refinement_level)

    # scheme = 'CR'
    if scheme == 'CR':
        # print 'we use Crouzieux-Raviart elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    else:
        V = dolfin.VectorFunctionSpace(mesh, "CG", vdgree)
        Q = dolfin.FunctionSpace(mesh, "CG", pdgree)

    if bccontrol:
        def _csf(s, nvec):
            return ((1. - 0.5*(1 + np.sin(s*2*np.pi + 0.5*np.pi)))*nvec[0],
                    (1. - 0.5*(1 + np.sin(s*2*np.pi + 0.5*np.pi)))*nvec[1])

        class ContBoundaryOne(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                dx = x[0] - xcenter
                dy = x[1] - ycenter
                r = dolfin.sqrt(dx*dx + dy*dy)
                inbbx = insidebbox(x, whichbox=1)
                return on_boundary and r < radius + bmarg and inbbx

        def cont_shape_one(element=None):
            class ContShapeOne(dolfin.UserExpression):

                def eval(self, value, x):
                    xvec = x - centvec.flatten()
                    aang = np.arccos(np.dot(xvec, b1base)
                                     / (np.linalg.norm(xvec)
                                        * np.linalg.norm(b1base)))
                    s = aang/extensrad

                    vls = _csf(s, b1normal)
                    value[0], value[1] = vls[0], vls[1]
                    if verbose:
                        dx = x[0] - xcenter
                        dy = x[1] - ycenter
                        r = dolfin.sqrt(dx*dx + dy*dy)
                        print(x - centvec.flatten(), ': s=', s, ': r=', r,
                              ':', np.linalg.norm(np.array(vls)))

                def value_shape(self):
                    return (2,)

            return ContShapeOne(element=element)

        class ContBoundaryTwo(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                dx = x[0] - xcenter
                dy = x[1] - ycenter
                r = dolfin.sqrt(dx*dx + dy*dy)
                inbbx = insidebbox(x, whichbox=2)
                return on_boundary and r < radius + bmarg and inbbx

        def cont_shape_two(element=None):
            class ContShapeTwo(dolfin.UserExpression):
                def eval(self, value, x):
                    xvec = x - centvec.flatten()
                    aang = np.arccos(np.dot(xvec, b2base)
                                     / (np.linalg.norm(xvec)
                                        * np.linalg.norm(b2base)))
                    s = aang/extensrad

                    vls = _csf(s, b2normal)
                    value[0], value[1] = vls[0], vls[1]
                    if verbose:
                        dx = x[0] - xcenter
                        dy = x[1] - ycenter
                        r = dolfin.sqrt(dx*dx + dy*dy)
                        print(x - centvec.flatten(), ': s=', s, ': r=', r,
                              ':', np.linalg.norm(np.array(vls)))

                def value_shape(self):
                    return (2,)
            return ContShapeTwo(element=element)

        bcsubdoms = [ContBoundaryOne, ContBoundaryTwo]
        bcshapefuns = [cont_shape_one(element=V.ufl_element()),
                       cont_shape_two(element=V.ufl_element())]
    else:
        bcsubdoms = [None, None]
        bcshapefuns = [None, None]

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
                           ymax=ymax, element=V.ufl_element())
    bc0 = dolfin.DirichletBC(V, g0, InflowBoundary())

    # Create no-slip boundary condition
    g1 = dolfin.Constant((0, 0))
    bc1 = dolfin.DirichletBC(V, g1, NoslipChannelWalls())

    # Create no-slip at cylinder surface
    bc1cyl = dolfin.DirichletBC(V, g1, NoslipCylinderSurface())

    # Create outflow boundary condition for pressure
    g2 = dolfin.Constant(0)
    bc2 = dolfin.DirichletBC(Q, g2, OutflowBoundary())

    # Collect boundary conditions
    bcu = [bc0, bc1, bc1cyl]
    bcp = [bc2]

    cylfems = dict(V=V,
                   Q=Q,
                   diribcs=bcu,
                   dirip=bcp,
                   contrbcssubdomains=bcsubdoms,
                   contrbcsshapefuns=bcshapefuns,
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


def cyl3D_fems(refinement_level=2, scheme='TH',
               bccontrol=False, verbose=False):
    """
    dictionary for the fem items for the 3D cylinder wake

    which is
     * the 2D setup extruded in z-direction
     * with symmetry BCs at the z-walls

    Parameters
    ----------
    scheme : {None, 'CR', 'TH'}
        the finite element scheme to be applied, 'CR' for Crouzieux-Raviart,\
        'TH' for Taylor-Hood, overrides `pdgree`, `vdgree`, defaults to `None`
    bccontrol : boolean, optional
        whether to consider boundary control via penalized Robin \
        defaults to `False`

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
                 domain of observation
         * `cdcoo`: dictionary with the coordinates of the domain of control
         * `uspacedep`: int that specifies in what spatial direction Bu \
                changes. The remaining is constant
         * `bcsubdoms`: list of subdomains that define the segments where \
                 the boundary control is applied

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
    # xmin = 0.0
    # xmax = 8.0
    # ymin = 0.0
    ymax = 1.5
    # zmin = 0.0
    # zmax = 1.0
    # xcenter = 2.0
    # ycenter = ymax/2
    # radius = 0.15

    # Load mesh
    mesh = dolfin.\
        Mesh("mesh/3d-cyl/karman3D_lvl{0}.xml.gz".format(refinement_level))

    # scheme = 'CR'
    if scheme == 'CR':
        # print 'we use Crouzieux-Raviart elements !'
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    elif scheme == 'TH':
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
        Q = dolfin.FunctionSpace(mesh, "CG", 1)

    # get the boundaries from the gmesh file
    meshfile = 'mesh/3d-cyl/karman3D_lvl{0}_facet_region.xml.gz'.\
        format(refinement_level)
    boundaries = dolfin.\
        MeshFunction('size_t', mesh, meshfile)

    # # Create inflow boundary condition
    # if no-slip at front and back
    # gin = dolfin.Expression(('6*(x[1]*(ymax-x[1]))/(ymax*ymax) * ' +
    #                          '6*(x[2]*(zmax-x[2]))/(zmax*zmax)',
    #                          '0.0', '0.0'),
    #                         ymax=ymax, zmax=zmax, element=V.ufl_element())
    gin = dolfin.Expression(('6*(x[1]*(ymax-x[1]))/(ymax*ymax)', '0.0', '0.0'),
                            ymax=ymax, element=V.ufl_element())
    bcin = dolfin.DirichletBC(V, gin, boundaries, 1)

    # ## Create no-slip boundary condition
    gzero = dolfin.Constant((0, 0, 0))

    # channel walls
    bcbt = dolfin.DirichletBC(V, gzero, boundaries, 2)  # bottom
    bctp = dolfin.DirichletBC(V, gzero, boundaries, 6)  # top

    # yes-slip at front and back -- only z-component set to zero
    gscalzero = dolfin.Constant(0)
    bcbc = dolfin.DirichletBC(V.sub(2), gscalzero, boundaries, 5)  # back
    bcfr = dolfin.DirichletBC(V.sub(2), gscalzero, boundaries, 4)  # front

    # Create no-slip at cylinder surface
    bccuc = dolfin.DirichletBC(V, gzero, boundaries, 9)  # uncontrolled
    bccco = dolfin.DirichletBC(V, gzero, boundaries, 7)  # ctrl upper
    bccct = dolfin.DirichletBC(V, gzero, boundaries, 8)  # ctrl lower

    # Create outflow boundary condition for pressure
    g2 = dolfin.Constant(0)
    bc2 = dolfin.DirichletBC(Q, g2, boundaries, 3)

    # Collect boundary conditions
    bcu = [bcin, bcbt, bcfr, bcbc, bctp, bccuc, bccco, bccct]
    bcp = [bc2]

    # Create right-hand side function
    fv = dolfin.Constant((0, 0, 0))
    fp = dolfin.Constant(0)

    def initial_conditions(self, V, Q):
        u0 = dolfin.Constant((0, 0, 0))
        p0 = dolfin.Constant(0)
        return u0, p0

    cylfems = dict(V=V,
                   Q=Q,
                   diribcs=bcu,
                   dirip=bcp,
                   # contrbcssubdomains=bcsubdoms,
                   # contrbcsshapefuns=bcshapefuns,
                   fv=fv,
                   fp=fp,
                   uspacedep=0,
                   charlen=0.3,
                   mesh=mesh)

    return cylfems


def gen_bccont_fems(scheme='TH', bccontrol=True, verbose=False,
                    strtomeshfile='', strtophysicalregions='',
                    inflowvel=1., inflowprofile='parabola',
                    movingwallcntrl=False,
                    strtobcsobs=''):
    """
    dictionary for the fem items for a general 2D flow setup

    with
     * inflow/outflow
     * boundary control

    Parameters
    ----------
    scheme : {None, 'CR', 'TH'}
        the finite element scheme to be applied, 'CR' for Crouzieux-Raviart,\
        'TH' for Taylor-Hood, overrides `pdgree`, `vdgree`, defaults to `None`
    bccontrol : boolean, optional
        whether to consider boundary control via penalized Robin \
        defaults to `True`
    movingwallcntrl : boolean, optional
        whether control is via moving boundaries

    Returns
    -------
    femp : a dictionary with the keys:
         * `V`: FEM space of the velocity
         * `Q`: FEM space of the pressure
         * `diribcs`: list of the (Dirichlet) boundary conditions
         * `dbcsinds`: list vortex indices with (Dirichlet) boundary conditions
         * `dbcsvals`: list of values of the (Dirichlet) boundary conditions
         * `dirip`: list of the (Dirichlet) boundary conditions \
                 for the pressure
         * `fv`: right hand side of the momentum equation
         * `fp`: right hand side of the continuity equation
         * `charlen`: characteristic length of the setup
         * `odcoo`: dictionary with the coordinates of the \
                 domain of observation

    """

    # Load mesh
    mesh = dolfin.Mesh(strtomeshfile)

    if scheme == 'CR':
        V = dolfin.VectorFunctionSpace(mesh, "CR", 1)
        Q = dolfin.FunctionSpace(mesh, "DG", 0)
    elif scheme == 'TH':
        V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
        Q = dolfin.FunctionSpace(mesh, "CG", 1)

    boundaries = dolfin.MeshFunction('size_t', mesh, strtophysicalregions)

    with open(strtobcsobs) as f:
        cntbcsdata = json.load(f)

    inflowgeodata = cntbcsdata['inflow']
    inflwpe = inflowgeodata['physical entity']
    inflwin = np.array(inflowgeodata['inward normal'])
    inflwxi = np.array(inflowgeodata['xone'])
    inflwxii = np.array(inflowgeodata['xtwo'])

    leninflwb = np.linalg.norm(inflwxi-inflwxii)

    if inflowprofile == 'block':
        inflwprfl = dolfin.\
            Expression(('cv*no', 'cv*nt'), cv=inflowvel,
                       no=inflwin[0], nt=inflwin[1],
                       element=V.ufl_element())
    elif inflowprofile == 'parabola':
        inflwprfl = InflowParabola(degree=2, lenb=leninflwb, xone=inflwxi,
                                   normalvec=inflwin, inflowvel=inflowvel)
    bcin = dolfin.DirichletBC(V, inflwprfl, boundaries, inflwpe)
    diribcu = [bcin]

    # ## THE WALLS
    wallspel = cntbcsdata['walls']['physical entity']
    gzero = dolfin.Constant((0, 0))
    for wpe in wallspel:
        diribcu.append(dolfin.DirichletBC(V, gzero, boundaries, wpe))
        bcdict = diribcu[-1].get_boundary_values()

    if not bccontrol:  # treat the control boundaries as walls
        try:
            for cntbc in cntbcsdata['controlbcs']:
                diribcu.append(dolfin.DirichletBC(V, gzero, boundaries,
                                                  cntbc['physical entity']))
        except KeyError:
            pass  # no control boundaries

    mvwdbcs = []
    mvwtvs = []
    try:
        for cntbc in cntbcsdata['moving walls']:
            center = np.array(cntbc['geometry']['center'])
            radius = cntbc['geometry']['radius']
            if cntbc['type'] == 'circle':
                omega = 1. if movingwallcntrl else 0.
                rotcyl = RotatingCircle(degree=2, radius=radius,
                                        xcenter=center, omega=omega)
            else:
                raise NotImplementedError()
            mvwdbcs.append(dolfin.DirichletBC(V, rotcyl, boundaries,
                                              cntbc['physical entity']))
    except KeyError:
        pass  # no moving walls defined
    if not movingwallcntrl:
        diribcu.extend(mvwdbcs)  # add the moving walls to the diri bcs
        mvwdbcs = []

    # Create outflow boundary condition for pressure
    # TODO XXX why zero pressure?? is this do-nothing???
    outflwpe = cntbcsdata['outflow']['physical entity']
    g2 = dolfin.Constant(0)
    bc2 = dolfin.DirichletBC(Q, g2, boundaries, outflwpe)

    # Collect boundary conditions
    bcp = [bc2]

    # Create right-hand side function
    fv = dolfin.Constant((0, 0))
    fp = dolfin.Constant(0)

    def initial_conditions(self, V, Q):
        u0 = dolfin.Constant((0, 0))
        p0 = dolfin.Constant(0)
        return u0, p0

    dbcinds, dbcvals = [], []
    for bc in diribcu:
        bcdict = bc.get_boundary_values()
        dbcvals.extend(list(bcdict.values()))
        dbcinds.extend(list(bcdict.keys()))

    mvwbcinds, mvwbcvals = [], []
    for bc in mvwdbcs:
        bcdict = bc.get_boundary_values()
        mvwbcvals.extend(list(bcdict.values()))
        mvwbcinds.extend(list(bcdict.keys()))

    # ## Control boundaries
    bcpes, bcshapefuns, bcds = [], [], []
    if bccontrol:
        for cbc in cntbcsdata['controlbcs']:
            cpe = cbc['physical entity']
            cxi, cxii = np.array(cbc['xone']), np.array(cbc['xtwo'])
            csf = _get_cont_shape_fun2D(xi=cxi, xii=cxii,
                                        element=V.ufl_element())
            bcshapefuns.append(csf)
            bcpes.append(cpe)
            bcds.append(dolfin.Measure("ds", subdomain_data=boundaries)(cpe))

    # ## Lift Drag Computation
    try:
        ldsurfpe = cntbcsdata['lift drag surface']['physical entity']
        liftdragds = dolfin.Measure("ds", subdomain_data=boundaries)(ldsurfpe)
        bclds = dolfin.DirichletBC(V, gzero, boundaries, ldsurfpe)
        bcldsdict = bclds.get_boundary_values()
        ldsbcinds = list(bcldsdict.keys())
    except KeyError:
        liftdragds = None  # no domain specified for lift/drag
        ldsbcinds = None
    try:
        outflwpe = cntbcsdata['outflow']['physical entity']
        outflowds = dolfin.Measure("ds", subdomain_data=boundaries)(outflwpe)
    except KeyError:
        outflowds = None  # no domain specified for outflow

    try:
        odcoo = cntbcsdata['observation-domain-coordinates']
    except KeyError:
        odcoo = None

    gbcfems = dict(V=V,
                   Q=Q,
                   dbcinds=dbcinds,
                   dbcvals=dbcvals,
                   mvwbcinds=mvwbcinds,
                   mvwbcvals=mvwbcvals,
                   mvwtvs=mvwtvs,
                   dirip=bcp,
                   outflowds=outflowds,
                   # contrbcssubdomains=bcsubdoms,
                   liftdragds=liftdragds,
                   ldsbcinds=ldsbcinds,
                   contrbcmeshfunc=boundaries,
                   contrbcspes=bcpes,
                   contrbcsshapefuns=bcshapefuns,
                   cntrbcsds=bcds,
                   odcoo=odcoo,
                   fv=fv,
                   fp=fp,
                   charlen=cntbcsdata['characteristic length'],
                   mesh=mesh)

    return gbcfems


def _get_cont_shape_fun2D(xi=None, xii=None, element=None, shape='parabola'):
    lencb = np.linalg.norm(xi-xii)
    cbt = 1./lencb*(xii-xi)  # the normalized vector pointing x1 -> x2
    cbn = np.array([cbt[1], -cbt[0]]).reshape((2, 1))  # rotate by pi/2

    class GenContShape(dolfin.UserExpression):

        def __init__(self, degree=2, element=None):
            self.degree = degree
            self.element = element
            super().__init__()

        def eval(self, value, x):
            curs = np.linalg.norm(x - xi)/lencb
            # print(x, curs)
            curvel = 6*curs*(1-curs)*cbn
            value[0], value[1] = curvel[0], curvel[1]

        def value_shape(self):
            return (2,)

    return GenContShape(element=element)


class InflowParabola(dolfin.UserExpression):
    '''Create inflow boundary condition

    a parabola g with `int g(s)ds = s1-s0 == int 1 ds`
    if on [0,1]: `g(s) = s*(1-s)*4*3/2`
    if on [s0,s1]: `g(s) = ((s-s0)/(s1-s0))*(1-(s-s0)/(s1-s0))*6`
    since then `g((s0+s1)/2)=3/2` and  `g(s0)=0=g(s1)`'''

    def __init__(self, degree=2, lenb=None, xone=None,
                 inflowvel=1., normalvec=None):
        self.degree = degree
        self.lenb = lenb
        self.xone = xone
        self.normalvec = normalvec
        self.inflowvel = inflowvel
        try:
            super().__init__()
        except RuntimeError():
            pass  # had trouble with this call to __init__ in 'dolfin:2017.2.0'

    def eval(self, value, x):
        curs = np.linalg.norm(x - self.xone)/self.lenb
        # print(x, curs)
        curvel = self.inflowvel*6*curs*(1-curs)*self.normalvec
        value[0], value[1] = curvel[0], curvel[1]

    def value_shape(self):
        return (2,)


class RotatingCircle(dolfin.UserExpression):
    '''Create the boundary condition of a rotating circle

    returns the angular velocity at the circle boundary
    '''

    def __init__(self, degree=2, radius=None, xcenter=None,
                 omega=1.):
        self.degree = degree
        self.radius = radius
        self.xcenter = xcenter
        self.anglevel = radius*omega
        print('Rotating cylinder: omega set to {0}'.format(omega))
        super().__init__()

    def eval(self, value, x):
        curn = 1./self.radius*(x - self.xcenter)
        # print(np.linalg.norm(curn))
        value[0], value[1] = -self.anglevel*curn[1], self.anglevel*curn[0]

    def value_shape(self):
        return (2,)


class LiftDragSurfForce():

    def __init__(self, V=None, nu=None, ldds=None,
                 outflowds=None, phione=None, phitwo=None):
        self.mesh = V.mesh()
        self.n = dolfin.FacetNormal(self.mesh)
        self.I = dolfin.Identity(self.mesh.geometry().dim())
        self.ldds = ldds
        self.outflowds = outflowds
        self.nu = nu
        self.A = dolfin.as_matrix([[0., 1.],
                                   [-1., 0.]])
        pickx = dolfin.as_matrix([[1., 0.], [0., 0.]])
        picky = dolfin.as_matrix([[0., 0.], [0., 1.]])
        self.phionex = pickx*phione
        self.phioney = picky*phione
        self.phitwo = phitwo

        def epsilon(u):
            return 0.5*(dolfin.nabla_grad(u) + dolfin.nabla_grad(u).T)

        self.epsilon = epsilon

    def evaliftdragforce(self, u=None, p=None):
        # ## The direct way
        # ux = u.sub(0)
        # uy = u.sub(1)
        # D = (2*self.nu*inner(self.epsilon(ux), dolfin.grad(self.phitwo))
        #      # inner(self.nu*grad(ux), grad(self.phione))
        #      + inner(u, grad(ux))*self.phione
        #      - p*self.phione.dx(0))*dolfin.dx
        # L = (inner(self.nu*grad(uy), grad(self.phione))
        #      + inner(u, grad(uy))*self.phione
        #      - p*self.phione.dx(1))*dolfin.dx
        # T = -p*self.I + 2.0*self.nu*self.epsilon(u)
        # force = dolfin.dot(T, self.n)
        # D = force[0]*self.ldds
        # L = force[1]*self.ldds
        # drag = dolfin.assemble(L)
        # lift = dolfin.assemble(D)

        inner, dx = dolfin.inner, dolfin.dx
        epsi, grad, div = self.epsilon, dolfin.grad, dolfin.div

        # ## the Babuska/Millner trick
        pox = self.phionex

        # convection
        drgo = inner(dolfin.dot(u, dolfin.nabla_grad(u)), pox)*dx
        # outflow correction
        ofcor = (self.nu*inner(grad(u).T*self.n, pox))*self.outflowds
        # diffusion
        drgt = (2*self.nu*inner(epsi(u), grad(self.phionex)))*dx
        # pressure
        drgd = -p*div(self.phionex)*dx

        drag = dolfin.assemble(drgt+drgd+drgo-ofcor)

        lfto = inner(dolfin.dot(u, dolfin.nabla_grad(u)), self.phioney)*dx
        lftt = 2*self.nu*inner(self.epsilon(u), dolfin.grad(self.phioney))*dx
        lftd = (-p*dolfin.div(self.phioney))*dx
        # outflow correction
        ofcor = (self.nu*inner(grad(u).T*self.n, self.phioney))*self.outflowds
        lift = dolfin.assemble(lfto+lftt+lftd-ofcor)
        return lift, drag

    def evatorqueSphere2D(self, u=None, p=None):
        inner, dx, grad = dolfin.inner, dolfin.dx, dolfin.grad

        donto = inner(dolfin.dot(u, dolfin.nabla_grad(u)), self.phitwo)*dx
        dontt = 2*self.nu*inner(self.epsilon(u), dolfin.grad(self.phitwo))*dx
        dontd = (-p*dolfin.div(self.phitwo))*dx
        ofcor = (self.nu*inner(grad(u).T*self.n, self.phitwo))*self.outflowds

        tconv = dolfin.assemble(donto)
        tdiff = dolfin.assemble(dontt)
        tpres = dolfin.assemble(dontd)
        tofcr = dolfin.assemble(ofcor)
        trqthr = tconv + tdiff + tpres - tofcr

        return trqthr
