import dolfin
import numpy as np
import scipy.sparse as sps

from dolfin import dx, grad, div, inner

dolfin.parameters.linear_algebra_backend = "uBLAS"

__all__ = ['ass_convmat_asmatquad',
           'get_stokessysmats',
           'get_convmats',
           'setget_rhs',
           'get_curfv',
           'get_convvec',
           'condense_sysmatsbybcs',
           'condense_velmatsbybcs',
           'expand_vp_dolfunc']


def ass_convmat_asmatquad(W=None, invindsw=None):
    """ assemble the convection matrix H, so that N(v)v = H[v.v]

    for the inner nodes.

    Notes
    -----
    Implemented only for 2D problems

    """
    mesh = W.mesh()
    deg = W.ufl_element().degree()
    fam = W.ufl_element().family()

    V = dolfin.FunctionSpace(mesh, fam, deg)

    # this is very specific for V being a 2D VectorFunctionSpace
    invindsv = invindsw[::2]/2

    v = dolfin.TrialFunction(V)
    vt = dolfin.TestFunction(V)

    def _pad_csrmats_wzerorows(smat, wheretoput='before'):
        """add zero rows before/after each row

        """
        indpeter = smat.indptr
        auxindp = np.c_[indpeter, indpeter].flatten()
        if wheretoput == 'after':
            smat.indptr = auxindp[1:]
        else:
            smat.indptr = auxindp[:-1]

        smat._shape = (2*smat.shape[0], smat.shape[1])

        return smat

    def _shuff_mrg_csrmats(xm, ym):
        """shuffle merge csr mats [xxx],[yyy] -> [xyxyxy]

        """
        xm.indices = 2*xm.indices
        ym.indices = 2*ym.indices + 1
        xm._shape = (xm.shape[0], 2*xm.shape[1])
        ym._shape = (ym.shape[0], 2*ym.shape[1])
        return xm + ym

    nklist = []
    for i in invindsv:
    # for i in range(V.dim()):
        # iterate for the columns
        bi = dolfin.Function(V)
        bvec = np.zeros((V.dim(), ))
        bvec[i] = 1
        bi.vector()[:] = bvec

        nxi = dolfin.assemble(v * bi.dx(0) * vt * dx)
        nyi = dolfin.assemble(v * bi.dx(1) * vt * dx)

        rows, cols, values = nxi.data()
        nxim = sps.csr_matrix((values, cols, rows))
        nxim.eliminate_zeros()

        rows, cols, values = nyi.data()
        nyim = sps.csr_matrix((values, cols, rows))
        nyim.eliminate_zeros()

        nxyim = _shuff_mrg_csrmats(nxim, nyim)
        nxyim = nxyim[invindsv, :][:, invindsw]
        nyxxim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='after')
        nyxyim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='before')

        nklist.extend([nyxxim, nyxyim])

    hmat = sps.hstack(nklist, format='csc')
    return hmat


def get_stokessysmats(V, Q, nu=1):
    """ Assembles the system matrices for Stokes equation

    in mixed FEM formulation, namely

    .. math::

        \\begin{bmatrix} A & -J' \\\\ J & 0 \\end{bmatrix}\
        \\colon V \\times Q \\to V' \\times Q'

    as the discrete representation of

    .. math::

        \\begin{bmatrix} -\\Delta & \\text{grad} \\\\ \
        \\text{div} & 0 \\end{bmatrix}

    plus the velocity and pressure mass matrices

    for a given trial and test space W = V * Q
    not considering boundary conditions.

    :param V:
        Fenics VectorFunctionSpace for the velocity
    :param Q:
        Fenics FunctionSpace for the pressure
    :param nu:
        viscosity parameter - defaults to 1

    :return:
        a dictionary with the following keys:
            * ``M``: the mass matrix of the velocity space,
            * ``A``: the stiffness matrix \
                :math:` \\nu (\\nabla \\phi_i, \\nabla \\phi_j)`
            * ``JT``: the gradient matrix,
            * ``J``: the divergence matrix, and
            * ``MP``: the mass matrix of the pressure space

    """

    u = dolfin.TrialFunction(V)
    p = dolfin.TrialFunction(Q)
    v = dolfin.TestFunction(V)
    q = dolfin.TestFunction(Q)

    ma = inner(u, v) * dx
    mp = inner(p, q) * dx
    aa = nu * inner(grad(u), grad(v)) * dx
    grada = div(v) * p * dx
    diva = q * div(u) * dx

    # Assemble system
    M = dolfin.assemble(ma)
    A = dolfin.assemble(aa)
    Grad = dolfin.assemble(grada)
    Div = dolfin.assemble(diva)
    MP = dolfin.assemble(mp)

    # Convert DOLFIN representation to scipy arrays
    rows, cols, values = M.data()
    Ma = sps.csr_matrix((values, cols, rows))

    rows, cols, values = MP.data()
    MPa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = A.data()
    Aa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = Grad.data()
    JTa = sps.csr_matrix((values, cols, rows))

    rows, cols, values = Div.data()
    Ja = sps.csr_matrix((values, cols, rows))

    stokesmats = {'M': Ma,
                  'A': Aa,
                  'JT': JTa,
                  'J': Ja,
                  'MP': MPa}

    return stokesmats


def get_convmats(u0_dolfun=None, u0_vec=None, V=None, invinds=None,
                 diribcs=None):
    """returns the matrices related to the linearized convection

    where u_0 is the linearization point

    Returns
    -------
    N1 : (N,N) sparse matrix
        representing :math:`(u_0 \\cdot \\nabla )u`
    N2 : (N,N) sparse matrix
        representing :math:`(u \\cdot \\nabla )u_0`
    fv : (N,1) array
        representing :math:`(u_0 \\cdot \\nabla )u_0`

    See Also
    --------
    stokes_navier_utils.get_v_conv_conts : the convection contributions \
            reduced to the inner nodes

    """

    if u0_vec is not None:
        u0, p = expand_vp_dolfunc(vc=u0_vec, V=V, diribcs=diribcs,
                                  invinds=invinds)
    else:
        u0 = u0_dolfun

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    # Assemble system
    n1 = inner(grad(u) * u0, v) * dx
    n2 = inner(grad(u0) * u, v) * dx
    f3 = inner(grad(u0) * u0, v) * dx

    n1 = dolfin.assemble(n1)
    n2 = dolfin.assemble(n2)
    f3 = dolfin.assemble(f3)

    # Convert DOLFIN representation to scipy arrays
    rows, cols, values = n1.data()
    N1 = sps.csr_matrix((values, cols, rows))
    N1.eliminate_zeros()

    rows, cols, values = n2.data()
    N2 = sps.csr_matrix((values, cols, rows))
    N2.eliminate_zeros()

    fv = f3.array()
    fv = fv.reshape(len(fv), 1)

    return N1, N2, fv


def setget_rhs(V, Q, fv, fp, t=None):

    if t is not None:
        fv.t = t
        fp.t = t
    elif hasattr(fv, 't') or hasattr(fp, 't'):
        Warning('No value for t specified')

    v = dolfin.TestFunction(V)
    q = dolfin.TestFunction(Q)

    fv = inner(fv, v) * dx
    fp = inner(fp, q) * dx

    fv = dolfin.assemble(fv)
    fp = dolfin.assemble(fp)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    fp = fp.array()
    fp = fp.reshape(len(fp), 1)

    rhsvecs = {'fv': fv,
               'fp': fp}

    return rhsvecs


def get_curfv(V, fv, invinds, tcur):
    """get the fv at innernotes at t=tcur

    """

    v = dolfin.TestFunction(V)

    fv.t = tcur

    fv = inner(fv, v) * dx

    fv = dolfin.assemble(fv)

    fv = fv.array()
    fv = fv.reshape(len(fv), 1)

    return fv[invinds, :]


def get_convvec(u0_dolfun=None, V=None, u0_vec=None, femp=None):
    """return the convection vector e.g. for explicit schemes

    given a dolfin function or the coefficient vector
    """

    if u0_vec is not None:
        u0, p = expand_vp_dolfunc(vc=u0_vec, V=V, diribcs=femp['diribcs'],
                                  invinds=femp['invinds'])
    else:
        u0 = u0_dolfun

    v = dolfin.TestFunction(V)
    ConvForm = inner(grad(u0) * u0, v) * dx

    ConvForm = dolfin.assemble(ConvForm)
    ConvVec = ConvForm.array()
    ConvVec = ConvVec.reshape(len(ConvVec), 1)

    return ConvVec


def condense_sysmatsbybcs(stms, velbcs):
    """resolve the Dirichlet BCs and condense the system matrices

    to the inner nodes

    Parameters
    ----------
    stms: dict
        of the stokes matrices with the keys
         * ``M``: the mass matrix of the velocity space,
         * ``A``: the stiffness matrix,
         * ``JT``: the gradient matrix,
         * ``J``: the divergence matrix, and
         * ``MP``: the mass matrix of the pressure space
    velbcs : list
        of dolfin Dirichlet boundary conditions for the velocity

    Returns
    -------
    stokesmatsc : dict
        a dictionary of the condensed matrices:
         * ``M``: the mass matrix of the velocity space,
         * ``A``: the stiffness matrix,
         * ``JT``: the gradient matrix, and
         * ``J``: the divergence matrix
    rhsvecsb : dict
        a dictionary of the contributions of the boundary data to the rhs:
         * ``fv``: contribution to momentum equation,
         * ``fp``: contribution to continuity equation
    invinds : (N,) array
        vector of indices of the inner nodes
    bcinds : (K,) array
        vector of indices of the boundary nodes
    bcvals : (K,) array
        vector of the values of the boundary nodes
    """

    nv = stms['A'].shape[0]

    auxu = np.zeros((nv, 1))
    bcinds = []
    for bc in velbcs:
        bcdict = bc.get_boundary_values()
        auxu[bcdict.keys(), 0] = bcdict.values()
        bcinds.extend(bcdict.keys())

    # putting the bcs into the right hand sides
    fvbc = - stms['A'] * auxu    # '*' is np.dot for csr matrices
    fpbc = - stms['J'] * auxu

    # indices of the innernodes
    invinds = np.setdiff1d(range(nv), bcinds).astype(np.int32)

    # extract the inner nodes equation coefficients
    Mc = stms['M'][invinds, :][:, invinds]
    Ac = stms['A'][invinds, :][:, invinds]
    fvbc = fvbc[invinds, :]
    Jc = stms['J'][:, invinds]
    JTc = stms['JT'][invinds, :]

    bcvals = auxu[bcinds]

    stokesmatsc = {'M': Mc,
                   'A': Ac,
                   'JT': JTc,
                   'J': Jc}

    rhsvecsbc = {'fv': fvbc,
                 'fp': fpbc}

    return stokesmatsc, rhsvecsbc, invinds, bcinds, bcvals


def condense_velmatsbybcs(A, velbcs):
    """resolve the Dirichlet BCs, condense velocity related matrices

    to the inner nodes, and compute the rhs contribution
    This is necessary when, e.g., the convection matrix changes with time

    Parameters
    ----------
    A : (N,N) sparse matrix
        coefficient matrix for the velocity
    velbcs : list
        of dolfin *dolfin* Dirichlet boundary conditions for the velocity

    Returns
    -------
    Ac : (K, K) sparse matrix
        the condensed velocity matrix
    fvbc : (K, 1) array
        the contribution to the rhs of the momentum equation

    """

    nv = A.shape[0]

    auxu = np.zeros((nv, 1))
    bcinds = []
    for bc in velbcs:
        bcdict = bc.get_boundary_values()
        auxu[bcdict.keys(), 0] = bcdict.values()
        bcinds.extend(bcdict.keys())

    # putting the bcs into the right hand sides
    fvbc = - A * auxu    # '*' is np.dot for csr matrices

    # indices of the innernodes
    invinds = np.setdiff1d(range(nv), bcinds).astype(np.int32)

    # extract the inner nodes equation coefficients
    Ac = A[invinds, :][:, invinds]
    fvbc = fvbc[invinds, :]

    return Ac, fvbc


def expand_vp_dolfunc(V=None, Q=None, invinds=None, diribcs=None, vp=None,
                      vc=None, pc=None):
    """expand v [and p] to the dolfin function representation

    Parameters
    ----------
    V : dolfin.VectorFunctionSpace
        FEM space of the velocity
    Q : dolfin.FunctionSpace
        FEM space of the pressure
    invinds : (N,) array
        vector of indices of the velocity nodes
    diribcs : list
        of the (Dirichlet) velocity boundary conditions
    vp : (N+M,1) array, optional
        solution vector of velocity and pressure
    v : (N,1) array, optional
        solution vector of velocity
    p : (M,1) array, optional
        solution vector of pressure

    Returns
    -------
    v : dolfin.Function(V)
        velocity as function
    p : dolfin.Function(Q), optional
        pressure as function
    """

    if vp is not None:
        vc = vp[:len(invinds), :]
        pc = vp[len(invinds):, :]
        p = dolfin.Function(Q)
    elif pc is not None:
        p = dolfin.Function(Q)

    v = dolfin.Function(V)
    ve = np.zeros((V.dim(), 1))

    # fill in the boundary values
    for bc in diribcs:
        bcdict = bc.get_boundary_values()
        ve[bcdict.keys(), 0] = bcdict.values()

    ve[invinds] = vc

    if pc is not None:
        pe = np.vstack([pc, [0]])
        p.vector().set_local(pe)
    else:
        p = None

    v.vector().set_local(ve)

    return v, p


def get_dof_coors(V, invinds=None):

    # doflist = []
    # coorlist = []
    # for (i, cell) in enumerate(dolfin.cells(V.mesh())):
    #     # print "Global dofs associated with cell %d: " % i,
    #     # print V.dofmap().cell_dofs(i)
    #     # print "The Dof coordinates:",
    #     # print V.dofmap().tabulate_coordinates(cell)
    #     dofs = V.dofmap().cell_dofs(i)
    #     coors = V.dofmap().tabulate_coordinates(cell)
    #     # Cdofs = V.dofmap().cell_dofs(i)
    #     coorlist.append(coors)
    #     doflist.append(dofs)

    # dofar = np.hstack(doflist)
    # coorar = np.vstack(coorlist)

    # unidofs, uniinds = np.unique(dofar, return_index=True)

    coorfun = dolfin.Expression(('x[0]', 'x[1]'))
    coorfun = dolfin.interpolate(coorfun, V)

    xinds = V.sub(0).dofmap().dofs()
    yinds = V.sub(1).dofmap().dofs()

    xcoors = coorfun.vector().array()[xinds]
    ycoors = coorfun.vector().array()[yinds]
    coorfunvec = coorfun.vector().array()

    if invinds is not None:
        # check which innerinds are xinds
        chix = np.intersect1d(invinds, xinds)
        chiy = np.intersect1d(invinds, yinds)
        chixx = np.in1d(invinds, xinds)
        # x inner inds in a inner vector
        xinds = np.arange(len(chixx), dtype=np.int32)[chixx]
        yinds = np.arange(len(chixx), dtype=np.int32)[~chixx]
        xcoors = coorfunvec[chix]
        ycoors = coorfunvec[chiy]
        coorfunvec = coorfunvec[invinds]

    coors = np.vstack([xcoors, ycoors]).T

    return coors, xinds, yinds, coorfunvec
