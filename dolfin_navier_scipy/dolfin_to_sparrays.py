import dolfin
import numpy as np
import scipy.sparse as sps

from dolfin import dx, grad, div, inner

# try:
#     dolfin.parameters.linear_algebra_backend = "Eigen"
# except RuntimeError:
#     dolfin.parameters.linear_algebra_backend = "uBLAS"

__all__ = ['ass_convmat_asmatquad',
           'get_stokessysmats',
           'get_convmats',
           'setget_rhs',
           'get_curfv',
           'get_convvec',
           'condense_sysmatsbybcs',
           'condense_velmatsbybcs',
           'expand_vp_dolfunc',
           'expand_vecnbc_dolfunc',
           'append_bcs_vec',
           'mat_dolfin2sparse']


def _unroll_dlfn_dbcs(diribclist, bcinds=None, bcvals=None):
    if diribclist is None:
        try:
            urbcinds, urbcvals = [], []
            for k, cbci in enumerate(bcinds):
                urbcinds.extend(cbci)
                urbcvals.extend(bcvals[k])
        except TypeError:
            # it's not a list of lists
            urbcinds, urbcvals = bcinds, bcvals

    else:
        urbcinds, urbcvals = [], []
        for bc in diribclist:
            bcdict = bc.get_boundary_values()
            urbcvals.extend(list(bcdict.values()))
            urbcinds.extend(list(bcdict.keys()))
    return urbcinds, urbcvals


def append_bcs_vec(vvec, V=None, vdim=None,
                   invinds=None, diribcs=None, **kwargs):
    """ append given boundary conditions to a vector representing inner nodes

    """
    if vdim is None:
        vdim = V.dim()

    vwbcs = np.zeros((vdim, 1))

    # fill in the boundary values
    for bc in diribcs:
        bcdict = bc.get_boundary_values()
        vwbcs[list(bcdict.keys()), 0] = list(bcdict.values())

    vwbcs[invinds] = vvec

    return vwbcs


def mat_dolfin2sparse(A):
    """get the csr matrix representing an assembled linear dolfin form

    """
    try:
        return dolfin.as_backend_type(A).sparray()
    except RuntimeError:  # `dolfin <= 1.5+` with `'uBLAS'` support
        rows, cols, values = A.data()
        return sps.csr_matrix((values, cols, rows))


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

        # get the i-th basis function
        bi = dolfin.Function(V)
        bvec = np.zeros((V.dim(), ))
        bvec[np.int(i)] = 1
        bi.vector()[:] = bvec

        # assemble for the i-th basis function
        nxi = dolfin.assemble(v * bi.dx(0) * vt * dx)
        nyi = dolfin.assemble(v * bi.dx(1) * vt * dx)

        nxim = mat_dolfin2sparse(nxi)
        nxim.eliminate_zeros()

        nyim = mat_dolfin2sparse(nyi)
        nyim.eliminate_zeros()

        # resorting of the arrays and inserting zero columns
        nxyim = _shuff_mrg_csrmats(nxim, nyim)
        nxyim = nxyim[invindsv, :][:, invindsw]
        nyxxim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='after')
        nyxyim = _pad_csrmats_wzerorows(nxyim.copy(), wheretoput='before')

        # tile the arrays in horizontal direction
        nklist.extend([nyxxim, nyxyim])

    hmat = sps.hstack(nklist, format='csc')
    return hmat


def get_stokessysmats(V, Q, nu=None, bccontrol=False,
                      cbclist=None, cbshapefuns=None):
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

    Parameters
    ----------
    V : dolfin.VectorFunctionSpace
        Fenics VectorFunctionSpace for the velocity
    Q : dolfin.FunctionSpace
        Fenics FunctionSpace for the pressure
    nu : float, optional
        viscosity parameter - defaults to 1
    bccontrol : boolean, optional
        whether boundary control (via penalized Robin conditions)
        is applied, defaults to `False`
    cbclist : list, optional
        list of dolfin's Subdomain classes describing the control boundaries
    cbshapefuns : list, optional
        list of spatial shape functions of the control boundaries

    Returns
    -------
    stokesmats, dictionary
        a dictionary with the following keys:
            * ``M``: the mass matrix of the velocity space,
            * ``A``: the stiffness matrix \
                :math:`\\nu \\int_\\Omega (\\nabla \\phi_i, \\nabla \\phi_j)`
            * ``JT``: the gradient matrix,
            * ``J``: the divergence matrix, and
            * ``MP``: the mass matrix of the pressure space
            * ``Apbc``: (N, N) sparse matrix, \
                the contribution of the Robin conditions to `A` \
                :math:`\\nu \\int_\\Gamma (\\phi_i, \\phi_j)`
            * ``Bpbc``: (N, k) sparse matrix, the input matrix of the Robin \
                conditions :math:`\\nu \\int_\\Gamma (\\phi_i, g_k)`, \
                where :math:`g_k` is the shape function associated with the \
                j-th control boundary segment

    """

    u = dolfin.TrialFunction(V)
    p = dolfin.TrialFunction(Q)
    v = dolfin.TestFunction(V)
    q = dolfin.TestFunction(Q)

    if nu is None:
        nu = 1
        print('No viscosity provided -- we set `nu=1`')

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
    Ma = mat_dolfin2sparse(M)
    MPa = mat_dolfin2sparse(MP)
    Aa = mat_dolfin2sparse(A)
    JTa = mat_dolfin2sparse(Grad)
    Ja = mat_dolfin2sparse(Div)

    stokesmats = {'M': Ma,
                  'A': Aa,
                  'JT': JTa,
                  'J': Ja,
                  'MP': MPa}

    if bccontrol:
        amatrobl, bmatrobl = [], []
        mesh = V.mesh()
        for bc, bcfun in zip(cbclist, cbshapefuns):
            # get an instance of the subdomain class
            Gamma = bc()

            # bparts = dolfin.MeshFunction('size_t', mesh,
            #                              mesh.topology().dim() - 1)

            boundaries = dolfin.MeshFunction("size_t", mesh,
                                             mesh.topology().dim()-1)
            boundaries.set_all(0)
            Gamma.mark(boundaries, 1)

            ds = dolfin.Measure('ds', domain=mesh, subdomain_data=boundaries)

            # Gamma.mark(bparts, 0)

            # Robin boundary form
            arob = dolfin.inner(u, v) * ds(1)  # , subdomain_data=bparts)
            brob = dolfin.inner(v, bcfun) * ds(1)  # , subdomain_data=bparts)

            amatrob = dolfin.assemble(arob)  # , exterior_facet_domains=bparts)
            bmatrob = dolfin.assemble(brob)  # , exterior_facet_domains=bparts)

            amatrob = mat_dolfin2sparse(amatrob)
            amatrob.eliminate_zeros()
            amatrobl.append(amatrob)
            bmatrobl.append(bmatrob.get_local().reshape((V.dim(), 1)))

        amatrob = amatrobl[0]
        for amatadd in amatrobl[1:]:
            amatrob = amatrob + amatadd
        bmatrob = np.hstack(bmatrobl)

        stokesmats.update({'amatrob': amatrob, 'bmatrob': bmatrob})

    return stokesmats


def get_convmats(u0_dolfun=None, u0_vec=None, V=None, invinds=None,
                 dbcvals=None, dbcinds=None, diribcs=None):
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
                                  dbcvals=dbcvals, dbcinds=dbcinds,
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
    N1 = mat_dolfin2sparse(n1)
    N1.eliminate_zeros()

    N2 = mat_dolfin2sparse(n2)
    N2.eliminate_zeros()

    fv = f3.get_local()
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

    fv = fv.get_local()
    fv = fv.reshape(len(fv), 1)

    fp = fp.get_local()
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

    fv = fv.get_local()
    fv = fv.reshape(len(fv), 1)

    return fv[invinds, :]


def get_convvec(u0_dolfun=None, V=None, u0_vec=None, femp=None,
                diribcs=None, invinds=None):
    """return the convection vector e.g. for explicit schemes

    given a dolfin function or the coefficient vector
    """

    if u0_vec is not None:
        if femp is not None:
            diribcs = femp['diribcs']
            invinds = femp['invinds']

        u0, p = expand_vp_dolfunc(vc=u0_vec, V=V, diribcs=diribcs,
                                  invinds=invinds)
    else:
        u0 = u0_dolfun

    v = dolfin.TestFunction(V)
    ConvForm = inner(grad(u0) * u0, v) * dx

    ConvForm = dolfin.assemble(ConvForm)
    if invinds is not None:
        ConvVec = ConvForm.get_local()[invinds]
    else:
        ConvVec = ConvForm.get_local()
    ConvVec = ConvVec.reshape(len(ConvVec), 1)

    return ConvVec


def condense_sysmatsbybcs(stms, velbcs=None, dbcinds=None, dbcvals=None,
                          mergerhs=False, rhsdict=None, ret_unrolled=False):
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
    velbcs : list, optional
        of dolfin Dirichlet boundary conditions for the velocity
    dbcinds: list, optional
        indices of the Dirichlet boundary conditions
    dbcvals: list, optional
        values of the Dirichlet boundary conditions (as listed in `dbcinds`)

    Returns
    -------
    stokesmatsc : dict
        a dictionary of the condensed matrices:
         * ``M``: the mass matrix of the velocity space,
         * ``A``: the stiffness matrix,
         * ``JT``: the gradient matrix, and
         * ``J``: the divergence matrix
         * ``MP``: the mass matrix of the pressure space
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

    if velbcs is not None:
        bcinds, bcvals = [], []
        for bc in velbcs:
            bcdict = bc.get_boundary_values()
            bcvals.extend(list(bcdict.values()))
            bcinds.extend(list(bcdict.keys()))
    else:
        bcinds, bcvals = dbcinds, dbcvals

    nv = stms['A'].shape[0]
    auxu = np.zeros((nv, 1))
    auxu[bcinds, 0] = bcvals

    # putting the bcs into the right hand sides
    fvbc = - stms['A'] * auxu    # '*' is np.dot for csr matrices
    fpbc = - stms['J'] * auxu

    # indices of the innernodes
    invinds = np.setdiff1d(list(range(nv)), bcinds).astype(np.int32)

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
                   'J': Jc,
                   'MP': stms['MP']}

    if mergerhs:
        rhsvecsbc = {'fv': rhsdict['fv'][invinds, :] + fvbc,
                     'fp': rhsdict['fp'] + fpbc}
    else:
        rhsvecsbc = {'fv': fvbc,
                     'fp': fpbc}

    if ret_unrolled:
        return (Mc, Ac, JTc, Jc, stms['MP'], rhsvecsbc['fv'], rhsvecsbc['fp'],
                invinds)
    else:
        return stokesmatsc, rhsvecsbc, invinds, bcinds, bcvals


def condense_velmatsbybcs(A, velbcs=None, return_bcinfo=False,
                          dbcinds=None, dbcvals=None, columnsonly=False):
    """resolve the Dirichlet BCs, condense velocity related matrices

    to the inner nodes, and compute the rhs contribution
    This is necessary when, e.g., the convection matrix changes with time

    Parameters
    ----------
    A : (N,N) sparse matrix
        coefficient matrix for the velocity
    velbcs : list
        of dolfin *dolfin* Dirichlet boundary conditions for the velocity
    return_bcinfo : boolean, optional
        if `True` a dict with the inner and the boundary indices is returned, \
        defaults to `False`
    columnsonly : boolean, optional
        whether to only reduce the columns, defaults to `False`

    Returns
    -------
    Ac : (K, K) sparse matrix
        the condensed velocity matrix
    fvbc : (K, 1) array
        the contribution to the rhs of the momentum equation
    dict, on demand
        with the keys
         * ``ininds``: indices of the inner nodes
         * ``bcinds``: indices of the boundary nodes

    """

    bcinds, bcvals = _unroll_dlfn_dbcs(velbcs, bcinds=dbcinds, bcvals=dbcvals)

    nv = A.shape[0]
    auxu = np.zeros((nv, 1))
    auxu[bcinds, 0] = bcvals

    # putting the bcs into the right hand sides
    fvbc = - A * auxu    # '*' is np.dot for csr matrices

    # indices of the innernodes
    ininds = np.setdiff1d(list(range(nv)), bcinds).astype(np.int32)

    if columnsonly:
        Ac = A[:, ininds]
    else:
        # extract the inner nodes equation coefficients
        Ac = A[ininds, :][:, ininds]
        fvbc = fvbc[ininds, :]

    if return_bcinfo:
        return Ac, fvbc, dict(ininds=ininds, bcinds=bcinds)
    else:
        return Ac, fvbc


def expand_vp_dolfunc(V=None, Q=None, invinds=None,
                      dbcinds=None, dbcvals=None,
                      diribcs=None, zerodiribcs=False,
                      vp=None, vc=None, pc=None, ppin=-1, **kwargs):
    """expand v [and p] to the dolfin function representation

    Parameters
    ----------
    V : dolfin.VectorFunctionSpace
        FEM space of the velocity
    Q : dolfin.FunctionSpace
        FEM space of the pressure
    invinds : (N,) array
        vector of indices of the velocity nodes
    diribcs : list, optional
        of the (Dirichlet) velocity boundary conditions, \
    dbcinds: list, optional
        indices of the Dirichlet boundary conditions
    dbcvals: list, optional
        values of the Dirichlet boundary conditions (as listed in `dbcinds`)
    zerodiribcs : boolean, optional
        whether to simply apply zero boundary conditions,
        defaults to `False`
    vp : (N+M,1) array, optional
        solution vector of velocity and pressure
    vc : (N,1) array, optional
        solution vector of velocity
    pc : (M,1) array, optional
        solution vector of pressure
    ppin : {int, None}, optional
        which dof of `p` is used to pin the pressure, defaults to `-1`

    Returns
    -------
    v : dolfin.Function(V)
        velocity as function
    p : dolfin.Function(Q), optional
        pressure as function

    Notes:
    ------
    if no Dirichlet boundary data is given, it is assumed that
    `vc` already contains the bc

    See Also
    --------
    expand_vecnbc_dolfunc : for a scalar function with multiple bcs
    """

    if vp is not None:
        vc = vp[:len(invinds), :]
        pc = vp[len(invinds):, :]
        p = dolfin.Function(Q)
    elif pc is not None:
        p = dolfin.Function(Q)

    v = dolfin.Function(V)

    if vc.size > V.dim():
        raise UserWarning('The dimension of the vector must no exceed V.dim')
    elif diribcs is None and dbcinds is None or len(vc) == V.dim():
        # we assume that the boundary conditions are already contained in vc
        ve = vc
    else:
        ve = np.zeros((V.dim(), 1))
        # print('ve w/o bcvals: ', np.linalg.norm(ve))
        # fill in the boundary values
        if not zerodiribcs:
            urbcinds, urbcvals = _unroll_dlfn_dbcs(diribcs, bcinds=dbcinds,
                                                   bcvals=dbcvals)
            ve[urbcinds, 0] = urbcvals
            # print('ve with bcvals :', np.linalg.norm(ve))
            # print('norm of bcvals :', np.linalg.norm(bcvals))

        ve = ve.flatten()
        ve[invinds] = vc.flatten()

    if pc is not None:
        if ppin is None:
            pe = pc
        elif ppin == -1:
            pe = np.vstack([pc, [0]])
        elif ppin == 0:
            pe = np.vstack([[0], pc])
        else:
            raise NotImplementedError()
        p.vector().set_local(pe)
    else:
        p = None

    v.vector().set_local(ve)

    return v, p


def expand_vecnbc_dolfunc(V=None, vec=None,
                          bcindsl=None, bcvalsl=None,
                          diribcs=None, bcsfaclist=None,
                          invinds=None):
    """expand a function vector with changing boundary conditions

    the boundary conditions may not be disjoint, what is used to model
    spatial dependencies of a control at the boundary.

    Parameters
    ----------
    V : dolfin.FunctionSpace
        FEM space of the scalar
    invinds : (N,) array
        vector of indices of the velocity nodes
    vec : (N,1) array
        solution vector
    diribcs : list
        of boundary conditions
    bcsfaclist : list, optional
        of factors for the boundary conditions

    Returns
    -------
    dolfin.function
        of the vector values and the bcs

    """

    v = dolfin.Function(V)
    ve = np.zeros((V.dim(), 1))
    if bcsfaclist is None:
        try:
            bcsfaclist = [1]*len(diribcs)
        except TypeError:
            bcsfaclist = [1]*len(bcvalsl)

    # fill in the boundary values
    if diribcs is not None:
        if not len(bcsfaclist) == len(diribcs):
            raise Warning('length of lists of bcs and facs not matching')
        for k, bc in enumerate(diribcs):
            bcdict = bc.get_boundary_values()
            ve[list(bcdict.keys()), 0] +=\
                bcsfaclist[k]*np.array(list(bcdict.values()))
    else:
        if not len(bcsfaclist) == len(bcvalsl):
            raise Warning('length of lists of bcs and facs not matching')
        for k, cfac in enumerate(bcsfaclist):
            ve[bcindsl[k], 0] += cfac*np.array(bcvalsl[k])

    ve[invinds] = vec
    v.vector().set_local(ve)
    return v


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

    coorfun = dolfin.Expression(('x[0]', 'x[1]'), element=V.ufl_element())
    coorfun = dolfin.interpolate(coorfun, V)

    xinds = V.sub(0).dofmap().dofs()
    yinds = V.sub(1).dofmap().dofs()

    xcoors = coorfun.vector().get_local()[xinds]
    ycoors = coorfun.vector().get_local()[yinds]
    coorfunvec = coorfun.vector().get_local()

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
