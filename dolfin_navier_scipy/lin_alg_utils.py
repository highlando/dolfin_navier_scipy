import numpy as np
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

__all__ = ['app_prj_via_sadpnt',
           'apply_sqrt_fromright',
           'apply_invsqrt_fromright',
           'get_Sinv_smw',
           'solve_sadpnt_smw',
           'app_luinv_to_spmat',
           'comp_sqfnrm_factrd_diff',
           'comp_sqfnrm_factrd_lyap_res',
           'comp_sqfnrm_factrd_sum',
           'mm_dnssps']


def app_prj_via_sadpnt(amat=None, jmat=None, rhsv=None,
                       jmatT=None, umat=None, vmat=None,
                       transposedprj=False):
    """Apply a projection via solving a saddle point problem.

    The theory is as follows. Consider the projection

    .. math::

        P = I - M^{-1}J_1^T(J_1^TM^{-1}J_2)^{-1}J_2.

    Then :math:`Pv` can be obtained via

    .. math::

        A^{-1}\\begin{bmatrix} Pv \\\\ * \end{bmatrix} = \
        \\begin{bmatrix} Mv \\\\ 0 \end{bmatrix},

    where

    .. math::

        A := \\begin{bmatrix} M & J_1^T \\\\ J_2 & 0 \\end{bmatrix}.

    And :math:`P^Tv` can be obtained via

    .. math::

        A^{-T}\\begin{bmatrix} M^{-T}P^Tv \\\\ * \end{bmatrix} = \
        \\begin{bmatrix} v \\\\ 0 \end{bmatrix}.

    Parameters
    ----------
    amat : (N,N) sparse matrix
        left upper entry in the saddlepoint matrix
    jmat : (M,N) sparse matrix
        left lower entry
    jmatT : (N,M) sparse matrix, optional
        right upper entry, defaults to `jmat.T`
    rhsv : (N,K) ndarray
        array to be projected
    umat, vmat : (N,L), (L,N) ndarrays or sparse matrices, optional
        factored contribution to `amat`, default to `None`
    transposedprj : boolean
        whether to apply the transpose of the projection, defaults to `False`

    Returns
    -------
    , : (N,K) ndarray
        projected `rhsv`

    """

    if jmatT is None:
        jmatT = jmat.T
    if jmat is None:
        jmat = jmatT.T

    if transposedprj:
        return amat.T * solve_sadpnt_smw(amat=amat.T, jmat=jmatT.T,
                                         rhsv=rhsv, jmatT=jmat.T,
                                         )[:amat.shape[0], :]

    else:
        if umat is None and vmat is None:
            arhsv = amat * rhsv
        else:
            arhsv = amat * rhsv - \
                np.dot(umat, np.dot(vmat, rhsv))

        return solve_sadpnt_smw(amat=amat, jmat=jmat, rhsv=arhsv,
                                jmatT=jmatT)[:amat.shape[0], :]


def solve_sadpnt_smw(amat=None, jmat=None, rhsv=None,
                     jmatT=None, umat=None, vmat=None,
                     rhsp=None, sadlu=None,
                     return_alu=False,
                     decouplevp=False, solve_A=None,
                     symmetric=False, posdefinite=False,
                     cgtol=1e-8,
                     krylov=None, krpslvprms={}, krplsprms={}):
    """solve a saddle point system

    A - np.dot(U,V)    J.T  *  X   =   rhsv
    J                   0              rhsp
    by sparse direct solves

    Parameters
    ----------
    amat : (N,N) sparse matrix
        left upper entry in the saddlepoint matrix
    jmat : (M,N) sparse matrix
        left lower entry
    jmatT : (N,M) sparse matrix, optional
        right upper entry, defaults to `jmat.T`
    rhsv : (N,K) ndarray
        upper part of right hand side
    rhsp : (M,K) ndarray, optional
        lower part of right hand side, defaults to zero
    umat, vmat : (N,L), (L,N) ndarrays or sparse matrices, optional
        factored contribution to `amat`, default to `None`
    sadlu : callable f(v), optional
        returns the inverse of the sadpoint matrix applied to `v`, defaults
        to `None`
    return_alu : boolean, optional
        whether to return the lu factored sadpoint matrix

    Returns
    -------
    , : (N,K) ndarray
        projected `rhsv`
    , : f(v) callable, optional
        lu decomposition of the saddlepoint matrix

    """

    nnpp = jmat.shape[0]

    if jmatT is None:
        jmatT = jmat.T
    if jmat is None:
        jmat = jmatT.T

    if rhsp is None:
        rhsp = np.zeros((nnpp, rhsv.shape[1]))

    # TODO --> that's pretty roughly implemented
    if decouplevp:
        if not symmetric and posdefinite:
            raise NotImplementedError('non symmetric not implemented')
        if solve_A is None:
            raise NotImplementedError('need a routine that gives `A.-1*rhs`')

        def _invJAinvJTp(p):
            return jmat*solve_A(jmatT*p)

        iJAiJT = spsla.LinearOperator((nnpp, nnpp), matvec=_invJAinvJTp,
                                      dtype=np.float32)
        import krypy
        prhs = jmat*solve_A(rhsv) - rhsp
        pls = krypy.linsys.LinearSystem(iJAiJT, prhs,  # M=TODO,
                                        self_adjoint=True,
                                        positive_definite=posdefinite)
        p = krypy.linsys.Cg(pls, tol=cgtol).xk
        v = solve_A(rhsv - jmatT*p)

        return np.vstack([v, p])
    # <-- TODO

    if sadlu is None:
        sysm1 = sps.hstack([amat, jmatT], format='csr')
        sysm2 = sps.hstack([jmat, sps.csr_matrix((nnpp, nnpp))], format='csr')
        mata = sps.vstack([sysm1, sysm2], format='csr')
    else:
        mata = sadlu

    if sps.isspmatrix(rhsv):
        rhs = np.vstack([np.array(rhsv.todense()), rhsp])
    else:
        rhs = np.vstack([rhsv, rhsp])

    if umat is not None:
        vmate = sps.hstack([vmat, sps.csc_matrix((vmat.shape[0], nnpp))])
        umate = np.vstack([umat, np.zeros((nnpp, umat.shape[1]))])
    else:
        umate, vmate = None, None

    if return_alu and not krylov:
        return app_smw_inv(mata, umat=umate, vmat=vmate, rhsa=rhs,
                           return_alu=True)

    else:
        return app_smw_inv(mata, umat=umate, vmat=vmate, rhsa=rhs,
                           krylov=krylov, krpslvprms=krpslvprms,
                           krplsprms=krplsprms)


def apply_massinv(M, rhsa, output=None):
    """ Apply the inverse of mass or any other spd matrix

    to a rhs array
    TODO: by now just a wrapper for spsla.spsolve
    change e.g. to CG

    Parameters
    ----------
    M : (N,N) sparse matrix
        symmetric strictly positive definite
    rhsa : (N,K) ndarray array or sparse matrix
        array the inverse of M is to be applied to
    output : string, optional
        set to 'sparse' if rhsa has many zero columns
        to get the output as a sparse matrix

    Returns
    -------
    , : (N,K) ndarray or sparse matrix
        the inverse of `M` applied to `rhsa`

    """

    if output == 'sparse':
        colinds = rhsa.tocsr().indices
        colinds = np.unique(colinds)
        rhsa_cpy = rhsa.tolil()
        for col in colinds:
            rhsa_cpy[:, col] = np.atleast_2d(spsla.spsolve(M,
                                             rhsa_cpy[:, col])).T
        return rhsa_cpy

    else:
        mlusolve = spsla.factorized(M.tocsc())
        try:
            mirhs = np.copy(rhsa.todense())
        except AttributeError:
            mirhs = np.copy(rhsa)

        for ccol in range(mirhs.shape[1]):
            mirhs[:, ccol] = mlusolve(mirhs[:, ccol])

        return mirhs


def apply_sqrt_fromright(M, rhsa, output=None):
    """apply the sqrt of a mass matrix or other spd

    TODO: cases for dense and sparse INPUTS

    Parameters
    ----------
    M : (N,N) sparse matrix
        symmetric strictly positive definite
    rhsa : (K,N) ndarray array or sparse matrix
        array the inverse of M is to be applied to
    output : string, optional
        set to 'sparse' if rhsa has many zero rows
        to get the output as a sparse matrix

    Returns
    -------
    , : (N,K) ndarray or sparse matrix
        the sqrt of the inverse of `M` applied to `rhsa` from the left

    """
    if sps.isspmatrix(M):
        Z = scipy.linalg.cholesky(M.todense())
    else:
        Z = scipy.linalg.cholesky(M)
    # R = Z.T*Z  <-> R^-1 = Z^-1*Z.-T
    if output == 'sparse':
        return sps.csc_matrix(rhsa * Z)
    else:
        return np.dot(rhsa, Z)


def apply_invsqrt_fromright(M, rhsa, output=None):
    """apply the sqrt of the inverse of a mass matrix or other spd

    TODO: cases for dense and sparse INPUTS

    Parameters
    ----------
    M : (N,N) sparse matrix
        symmetric strictly positive definite
    rhsa : (K,N) ndarray array or sparse matrix
        array the inverse of M is to be applied to
    output : string, optional
        set to 'sparse' if rhsa has many zero rows
        to get the output as a sparse matrix

    Returns
    -------
    , : (N,K) ndarray or sparse matrix
        the sqrt of the inverse of `M` applied to `rhsa` from the left

    """
    try:
        Z = scipy.linalg.cholesky(M.todense())
    except AttributeError:
        Z = scipy.linalg.cholesky(M)
    # R = Z.T*Z  <-> R^-1 = Z^-1*Z.-T
    if output == 'sparse':
        return sps.csc_matrix(rhsa * np.linalg.inv(Z.T))
    else:
        return np.dot(rhsa, np.linalg.inv(Z.T))


def get_Sinv_smw(amat_lu, umat=None, vmat=None):
    """ compute inverse of I-V*Ainv*U as it is needed for

    the application of the Sherman-Morrison-Woodbury formula

    Parameters
    ----------
    amat_lu : callable f(v) or (N,N) sparse matrix
        the main part of the matrix `A-UV` possibly lu-factored
    umat, vmat : (N,L), (L,N) ndarrays or sparse matrices
        factored contribution to `amat_lu`

    Returns
    -------
    , : (L,L) ndarray
        small inverse in the smw update

    """
    # aiu = np.zeros(umat.shape)
    if sps.isspmatrix(umat):
        locumat = np.array(umat.todense())
    else:
        locumat = umat

    aiul = []  # to allow for complex values
    for ccol in range(locumat.shape[1]):
        try:
            aiul.append(amat_lu(locumat[:, ccol]))
        except TypeError:
            aiul.append(spsla.spsolve(amat_lu, locumat[:, ccol]))
    aiu = np.asarray(aiul).T

    if sps.isspmatrix(vmat):
        return np.linalg.inv(np.eye(umat.shape[1]) - vmat * aiu)
    else:
        return np.linalg.inv(np.eye(umat.shape[1]) - np.dot(vmat, aiu))


def app_luinv_to_spmat(alu_solve, Z):
    """ compute A.-1*Z  where A comes factored

    and with a solve routine for possibly complex Z

    Parameters
    ----------
    alu_solve : callable f(v)
        returning a matrix inverse applied to `v`
    Z : (N,K) ndarray, real or complex
        the inverse is to be applied to

    Returns
    -------
    , : (N,K) ndarray
        matrix inverse applied to ndarray

    """

    Z.tocsc()
    # ainvz = np.zeros(Z.shape)
    ainvzl = []  # to allow for complex values
    for ccol in range(Z.shape[1]):
        zcol = Z[:, ccol].toarray().flatten()
        if np.isrealobj(zcol):
            ainvzl.append(alu_solve(zcol))
        else:
            ainvzl.append(alu_solve(zcol.real) +
                          1j*alu_solve(zcol.imag))

    return np.asarray(ainvzl).T


def app_smw_inv(amat, umat=None, vmat=None, rhsa=None, Sinv=None,
                savefactoredby=5, return_alu=False, alu=None,
                krylov=None, krpslvprms={}, krplsprms={}):
    """compute the sherman morrison woodbury inverse

    of `A - np.dot(U,V)` applied to an (array)rhs.

    Parameters
    ----------
    amat : (N,N) sparse matrix
        main part of `A-UV`
    umat, vmat : (N,L), (L,N) ndarrays or sparse matrices, optional
        factored contribution to `amat`, default to `None`
    rhsa : (N,K) ndarray array or sparse matrix
        array the inverse of `A-UV` is to be applied to
    Sinv : (L,L) ndarray, optional
        the 'small' inverse in the smw formula, defaults to `None`
    savefactoredby : integer, optional
        if the number of columns of `rhsa` exceeds this parameter, the
        lu decomposition of `amat` is stored
    return_alu : boolean, optional
        whether to return the lu decomposition of `amat`, defaults to `False`
    alu : amat.factorized(), optional
        `lu` factorization of amat
    krylov : {None, 'gmres'}, optional
        whether or not to use an iterative solver, defaults to `None`
    krpslvprms : dictionary, optional
        to specify parameters of the linear solver for use in Krypy, e.g.,

          * 'x0', nparray: initial guess
          * tolerance
          * max number of iterations
          * 'convstatsl', list: for convergence statistics

        defaults to `None`
    krplsprms : dictionary, optional
        parameters to define the linear system like

          * preconditioner

    Returns
    -------
    , : (N,K) ndarray
        the inverse of `A-UV` applied to rhsa

    """
    if krylov is not None:
        import krypy.linsys as kls

        def auvb(v):
            if umat is None:
                return amat*v
            else:
                return amat*v - mm_dnssps(umat, mm_dnssps(vmat, v))

        auvblo = spsla.LinearOperator(amat.shape, matvec=auvb,
                                      dtype='float64')

        auvirhs = []
        try:
            citcl = []
            for rhscol in range(rhsa.shape[1]):
                crhs = rhsa[:, rhscol]
                krplinsys = kls.LinearSystem(A=auvblo, b=crhs, **krplsprms)
                solinst = kls.Gmres(krplinsys, **krpslvprms)
                auvirhs.append(solinst.xk)
                # solinst.xk = None  # strip off solution vector for stats
                citcl.append(solinst.resnorms)
            krpslvprms['convstatsl'].append(citcl)
        except KeyError:
            pass  # no stats

        return np.asarray(auvirhs)[:, :, 0].T

    else:
        if rhsa.shape[1] >= savefactoredby or return_alu:
            try:
                alu = spsla.factorized(amat)
            except (NotImplementedError, TypeError):
                alu = amat
        elif alu is None:
            alu = amat

        # auvirhs = np.zeros(rhsa.shape)
        auvirhs = []
        for rhscol in range(rhsa.shape[1]):
            crhs = rhsa[:, rhscol]
            # branch with u and v present
            if umat is not None:
                if Sinv is None:
                    Sinv = get_Sinv_smw(alu, umat, vmat)

                # the corrected rhs: (I + U*Sinv*V*Ainv)*rhs
                try:
                    # if Alu comes factorized, e.g. LU-factored - fine
                    aicrhs = alu(crhs)
                except TypeError:
                    aicrhs = spsla.spsolve(alu, crhs)

                if sps.isspmatrix(vmat):
                    crhs = crhs + mm_dnssps(umat, np.dot(Sinv, vmat * aicrhs))
                else:
                    crhs = crhs + mm_dnssps(umat,
                                            np.dot(Sinv, np.dot(vmat, aicrhs)))

            try:
                # auvirhs[:, rhscol] = alu(crhs)
                auvirhs.append(alu(crhs))
            except TypeError:
                # auvirhs[:, rhscol] = spsla.spsolve(alu, crhs)
                auvirhs.append(spsla.spsolve(alu, np.array(crhs)))

        if return_alu:
            return np.asarray(auvirhs).T, alu
        else:
            return np.asarray(auvirhs).T


def comp_sqfnrm_factrd_diff(zone, ztwo, ret_sing_norms=False):
    """compute the squared Frobenius norm of z1*z1.T - z2*z2.T

    using the linearity traces and that tr(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(zone.T, zone)
    btb = np.dot(ztwo.T, ztwo)
    atb = np.dot(zone.T, ztwo)

    if ret_sing_norms:
        norm_z1 = (ata * ata).sum(-1).sum()
        norm_z2 = (btb * btb).sum(-1).sum()
        return (norm_z1 - 2 * (atb * atb).sum(-1).sum() + norm_z2,
                norm_z1,
                norm_z2)

    return (ata * ata).sum(-1).sum() -  \
        2 * (atb * atb).sum(-1).sum() + \
        (btb * btb).sum(-1).sum()


def comp_sqfnrm_factrd_sum(zone, ztwo, ret_sing_norms=False):
    """compute the squared Frobenius norm of z1*z1.T + z2*z2.T

    using the linearity traces and that tr.(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(zone.T, zone)
    btb = np.dot(ztwo.T, ztwo)
    atb = np.dot(zone.T, ztwo)

    if ret_sing_norms:
        norm_z1 = (ata * ata).sum(-1).sum()
        norm_z2 = (btb * btb).sum(-1).sum()
        return (norm_z1 + 2 * (atb * atb).sum(-1).sum() + norm_z2,
                norm_z1,
                norm_z2)

    return (ata * ata).sum(-1).sum() +  \
        2 * (atb * atb).sum(-1).sum() + \
        (btb * btb).sum(-1).sum()


def comp_sqfnrm_factrd_lyap_res(A, B, C):
    """compute the squared Frobenius norm of A*B.T + B*A.T + C*C.T

    using the linearity traces and that tr.(z1.dot(z2)) = tr(z2.dot(z1))
    and that tr(z1.dot(z1.T)) is faster computed via (z1*z1.sum(-1)).sum()
    """

    ata = np.dot(A.T, A)
    atb = np.dot(A.T, B)
    atc = np.dot(A.T, C)
    btb = np.dot(B.T, B)
    btc = np.dot(B.T, C)
    ctc = np.dot(C.T, C)

    return 2 * (btb * ata).sum(-1).sum() +  \
        2 * (atb * atb.T).sum(-1).sum() + \
        4 * (btc.T * atc.T).sum(-1).sum() + \
        (ctc * ctc).sum(-1).sum()


def comp_uvz_spdns(umat, vmat, zmat, startleft=False):
    """comp u*[v*z] (default) or [u*v]*z for sparse or dense u or v

    `if startleft` we compute [u*v]*z
    """

    if startleft:
        return mm_dnssps(mm_dnssps(umat, vmat), zmat)
        # if sps.isspmatrix(vmat) or sps.isspmatrix(zmat):
        #     vz = vmat * zmat
        # else:
        #     vz = np.dot(vmat, zmat)
        # if sps.isspmatrix(umat):
        #     return umat * vz
        # else:
        #     return np.dot(umat, vz)
    else:
        return mm_dnssps(umat, mm_dnssps(vmat, zmat))
        # if sps.isspmatrix(umat) or sps.isspmatrix(vmat):
        #     uv = umat * vmat
        # else:
        #     uv = np.dot(umat, vmat)
        # if sps.isspmatrix(zmat):
        #     return uv * zmat
        # else:
        #     return np.dot(umat, vz)


def mm_dnssps(A, v):
    """compute A*v for sparse or dense A"""
    try:
        return A.matvec(v)
    except AttributeError:
        pass
    if sps.isspmatrix(A) or sps.isspmatrix(v):
        return A*v
    else:
        return np.dot(A, v)
