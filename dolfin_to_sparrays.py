import dolfin
import numpy as np
import scipy.sparse as sps

from dolfin import dx, grad, div, inner

dolfin.parameters.linear_algebra_backend = "uBLAS"

__all__ = ['get_stokessysmats',
           'get_convmats',
           'setget_rhs',
           'get_curfv',
           'get_convvec',
           'condense_sysmatsbybcs',
           'condense_velmatsbybcs',
           'expand_vp_dolfunc']


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

    :return:
        ``N1`` matrix representing :math:`(u_0 \\cdot \\nabla )u`
        ``N2`` matrix representing :math:`(u \\cdot \\nabla )u_0`
        ``fv`` vector representing :math:`(u_0 \\cdot \\nabla )u_0`

    at the inner nodes

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

    rows, cols, values = n2.data()
    N2 = sps.csr_matrix((values, cols, rows))

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

    :param stms:
        dictionary of the stokes matrices with the keys
        * ``M``: the mass matrix of the velocity space,
        * ``A``: the stiffness matrix,
        * ``JT``: the gradient matrix,
        * ``J``: the divergence matrix, and
        * ``MP``: the mass matrix of the pressure space

    :param velbcs:
        a list of dolfin Dirichlet boundary conditions for the velocity

    :return stokesmatsc:
        a dictionary of the condensed matrices:
        * ``M``: the mass matrix of the velocity space,
        * ``A``: the stiffness matrix,
        * ``JT``: the gradient matrix, and
        * ``J``: the divergence matrix

    :return rhsvecsbc:
        a dictionary of the contributions of the boundary data to the rhs:
        * ``fv``: contribution to momentum equation,
        * ``fp``: contribution to continuity equation

    :return invinds:
        vector of indices of the inner nodes

    :return bcinds:
        vector of indices of the boundary nodes

    :return bcvals:
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

    :param A:
        coefficient matrix for the velocity
    :param velbcs:
        a list of dolfin Dirichlet boundary conditions for the velocity

    :return Ac:
        the condensed velocity matrix
    :return fvbc:
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

    pdof = pressure dof that was set zero

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


def export_mats_to_matlab(M=None, A=None, J=None, matfname='matexport'):
    import scipy.io
    infostring = 'to give the (linearized) momentum eqn as' +\
                 'M\dot v + A v - J^T p = 0 '
    scipy.io.savemat(matfname, dict(M=M, A=A, J=J, B=B,
                                    infostring=infostring))
