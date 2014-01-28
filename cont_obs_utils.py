import dolfin
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import lin_alg_utils as lau

from dolfin import dx, inner

dolfin.parameters.linear_algebra_backend = "uBLAS"


def get_inp_opa(cdcoo=None, NU=8, V=None):
    """dolfin.assemble the 'B' matrix

    the findim array representation
    of the input operator """

    cdom = ContDomain(cdcoo)

    v = dolfin.TestFunction(V)
    v_one = dolfin.Expression(('1', '1'))
    v_one = dolfin.interpolate(v_one, V)

    BX, BY = [], []

    for nbf in range(NU):
        ubf = L2abLinBas(nbf, NU)
        bux = Cast1Dto2D(ubf, cdom, vcomp=0, xcomp=0)
        buy = Cast1Dto2D(ubf, cdom, vcomp=1, xcomp=0)
        bx = inner(v, bux) * dx
        by = inner(v, buy) * dx
        Bx = dolfin.assemble(bx)
        By = dolfin.assemble(by)
        Bx = Bx.array()
        By = By.array()
        Bx = Bx.reshape(len(Bx), 1)
        By = By.reshape(len(By), 1)
        BX.append(sps.csc_matrix(By))
        BY.append(sps.csc_matrix(Bx))

    Mu = ubf.massmat()

    return (
        sps.hstack([sps.hstack(BX), sps.hstack(BY)],
                   format='csc'), sps.block_diag([Mu, Mu])
    )


def get_mout_opa(odcoo=None, NY=8, V=None, NV=20):
    """dolfin.assemble the 'MyC' matrix

    the find an array representation
    of the output operator

    the considered output is y(s) = 1/C int_x[1] v(s,x[1]) dx[1]
    it is computed by testing computing my_i = 1/C int y_i v d(x)
    where y_i depends only on x[1] and y_i is zero outside the domain
    of observation. 1/C is the width of the domain of observation
    cf. doku
    """

    odom = ContDomain(odcoo)

    v = dolfin.TestFunction(V)
    voney = dolfin.Expression(('0', '1'))
    vonex = dolfin.Expression(('1', '0'))
    voney = dolfin.interpolate(voney, V)
    vonex = dolfin.interpolate(vonex, V)

    # factor to compute the average via \bar u = 1/h \int_0^h u(x) dx
    Ci = 1.0 / (odcoo['xmax'] - odcoo['xmin'])

    omega_y = dolfin.RectangleMesh(odcoo['xmin'], odcoo['ymin'],
                                   odcoo['xmax'], odcoo['ymax'],
                                   NV/5, NY-1)

    y_y = dolfin.VectorFunctionSpace(omega_y, 'CG', 1)
    # vone_yx = dolfin.interpolate(vonex, y_y)
    # vone_yy = dolfin.interpolate(voney, y_y)

    # charfun = CharactFun(odom)
    # v = dolfin.TestFunction(V)
    # checkf = dolfin.assemble(inner(v, charfun) * dx)
    # dofs_on_subd = np.where(checkf.array() > 0)[0]

    charfun = CharactFun(odom)
    v = dolfin.TestFunction(V)
    u = dolfin.TrialFunction(V)

    MP = dolfin.assemble(inner(v, u) * charfun * dx)

    rows, cols, values = MP.data()
    MPa = sps.dia_matrix(sps.csr_matrix((values, cols, rows)))

    checkf = MPa.diagonal()
    dofs_on_subd = np.where(checkf > 0)[0]

    # set up the numbers of zero columns to be inserted
    # between the nonzeros, i.e. the columns corresponding
    # to the dofs of V outside the observation domain
    # e.g. if there are 7 dofs and dofs_on_subd = [1,4,5], then
    # we compute the numbers [1,2,0,1] to assemble C = [0,*,0,0,*,*,0]
    indist_subddofs = dofs_on_subd[1:] - (dofs_on_subd[:-1]+1)
    indist_subddofs = np.r_[indist_subddofs, V.dim() - dofs_on_subd[-1] - 1]

    YX = [sps.csc_matrix((NY, dofs_on_subd[0]))]
    YY = [sps.csc_matrix((NY, dofs_on_subd[0]))]
    kkk = 0
    for curdof, curzeros in zip(dofs_on_subd, indist_subddofs):
        kkk += 1
        vcur = dolfin.Function(V)
        vcur.vector()[:] = 0
        vcur.vector()[curdof] = 1
        vdof_y = dolfin.interpolate(vcur, y_y)

        Yx, Yy = [], []
        for nbf in range(NY):
            ybf = L2abLinBas(nbf, NY, a=odcoo['ymin'], b=odcoo['ymax'])
            yx = Cast1Dto2D(ybf, odom, vcomp=0, xcomp=1)
            yy = Cast1Dto2D(ybf, odom, vcomp=1, xcomp=1)

            yxf = Ci * inner(vdof_y, yx) * dx
            yyf = Ci * inner(vdof_y, yy) * dx

            # if kkk < 3:
            # curvyx = inner(vdof_y, yx) * dx
            # curvyy = inner(vdof_y, yy) * dx
            # vdofonex = inner(vdof_y, vone_yx) * dx
            # vdofoney = inner(vdof_y, vone_yy) * dx
            # print 'DOF number {0}: {1}'.format(curdof,
            #                                    [dolfin.assemble(curvyx),
            #                                     dolfin.assemble(curvyy),
            #                                     dolfin.assemble(vdofonex),
            #                                     dolfin.assemble(vdofoney)])
            # if kkk == 3:
                # raise Warning('TODO: debug')

            Yx.append(dolfin.assemble(yxf))
            # ,
            #                      form_compiler_parameters={
            #                          'quadrature_rule': 'canonical',
            #                          'quadrature_degree': 2})
            Yy.append(dolfin.assemble(yyf))
            # ,
            #                      form_compiler_parameters={
            #                          'quadrature_rule': 'default',
            #                          'quadrature_degree': 2})

        Yx = np.array(Yx)
        Yy = np.array(Yy)
        Yx = Yx.reshape(NY, 1)
        Yy = Yy.reshape(NY, 1)
        # append the columns to z
        YX.append(sps.csc_matrix(Yx))
        YY.append(sps.csc_matrix(Yy))
        if curzeros > 0:
            YX.append(sps.csc_matrix((NY, curzeros)))
            YY.append(sps.csc_matrix((NY, curzeros)))

    # print 'number of subdofs: {0}'.format(dofs_on_subd.shape[0])
    My = ybf.massmat()
    YYX = sps.hstack(YX)
    YYY = sps.hstack(YY)
    MyC = sps.vstack([YYX, YYY], format='csc')

    # basfun = dolfin.Function(V)
    # basfun.vector()[dofs_on_subd] = 0.2
    # basfun.vector()[0] = 1  # for scaling the others only
    # dolfin.plot(basfun)

    return (MyC, sps.block_diag([My, My], format='csc'))


def app_difffreeproj(v=None, J=None, M=None):
    """apply the regularization (projection to divfree vels)

    i.e. compute v = [I-M^-1*J.T*S^-1*J]v
    """

    vg = lau.app_schurc_inv(M, J, np.atleast_2d(J * v).T)

    vg = spsla.spsolve(M, J.T * vg)

    return v - vg


def get_regularized_c(Ct=None, J=None, Mt=None):
    """apply the regularization (projection to divfree vels)

    i.e. compute rC = C*[I-M^-1*J.T*S^-1*J] as
    rCT = [I - J.T*S.-T*J*M.-T]*C.T
    """

    raise UserWarning('deprecated - use more explicit approach to proj via ' +\
                      'sadpoints systems as implemented in linalg_utils')

    Nv, NY = Mt.shape[0], Ct.shape[1]
    try:
        rCt = np.load('data/regCNY{0}vdim{1}.npy'.format(NY, Nv))
    except IOError:
        print 'no data/regCNY{0}vdim{1}.npy'.format(NY, Nv)
        MTlu = spsla.factorized(Mt)
        auCt = np.zeros(Ct.shape)
        # M.-T*C.T
        for ccol in range(NY):
            auCt[:, ccol] = MTlu(np.array(Ct[:, ccol].todense())[:, 0])
        # J*M.-T*C.T
        auCt = J * auCt
        # S.-T*J*M.-T*C.T
        auCt = lau.app_schurc_inv(MTlu, J, auCt)
        rCt = Ct - J.T * auCt
        np.save('data/regCNY{0}vdim{1}.npy'.format(NY, Nv), rCt)

    return np.array(rCt)


# Subdomains of Control and Observation
class ContDomain(dolfin.SubDomain):

    def __init__(self, ddict):
        dolfin.SubDomain.__init__(self)
        self.minxy = [ddict['xmin'], ddict['ymin']]
        self.maxxy = [ddict['xmax'], ddict['ymax']]

    def inside(self, x, on_boundary):
        return (dolfin.between(x[0], (self.minxy[0], self.maxxy[0]))
                and
                dolfin.between(x[1], (self.minxy[1], self.maxxy[1])))


class L2abLinBas():
    """ return the hat function related to the num-th vertex

    from the interval [a=0, b=1] with an equispaced grid
    of N vertices """

    def __init__(self, num, N, a=0.0, b=1.0):
        self.dist = (b - a) / (N - 1)
        self.vertex = a + num * self.dist
        self.num, self.N = num, N
        self.a, self.b = a, b

    def evaluate(self, s):
        # print s
        if max(self.a, self.vertex - self.dist) <= s <= self.vertex:
            sval = 1.0 - 1.0 / self.dist * (self.vertex - s)
        elif self.vertex <= s <= min(self.b, self.vertex + self.dist):
            sval = 1.0 - 1.0 / self.dist * (s - self.vertex)
        else:
            sval = 0
        return sval

    def massmat(self):
        """ return the mass matrix
        """
        mesh = dolfin.IntervalMesh(self.N - 1, self.a, self.b)
        Y = dolfin.FunctionSpace(mesh, 'CG', 1)
        yv = dolfin.TestFunction(Y)
        yu = dolfin.TrialFunction(Y)
        my = yv * yu * dx
        my = dolfin.assemble(my)
        rows, cols, values = my.data()
        return sps.csr_matrix((values, cols, rows))


class Cast1Dto2D(dolfin.Expression):
    """ casts a function u defined on [u.a, u.b]

    into the f[comp] of an expression
    defined on a 2D domain cdom by
    by scaling to fit the xcomp extension
    and simply extruding into the other direction
    """

    def __init__(self, u, cdom, vcomp=None, xcomp=0):
        # control 1D basis function
        self.u = u
        # domain of control
        self.cdom = cdom
        # component of the value to be set as u(s)
        self.vcomp = vcomp
        # component of x to be considered as s coordinate
        self.xcomp = xcomp
        # transformation of the intervals [cd.xmin, cd.xmax] -> [a, b]
        # via s = m*x + d
        self.m = (self.u.b - self.u.a) / \
            (cdom.maxxy[self.xcomp] - cdom.minxy[self.xcomp])
        self.d = self.u.b - self.m * cdom.maxxy[self.xcomp]

    def eval(self, value, x):
        if self.cdom.inside(x, False):
            if self.xcomp is None:
                value[:] = self.u.evaluate(self.m * x[self.xcomp] + self.d)
            else:
                value[:] = 0
                value[self.vcomp] = self.u.evaluate(
                    self.m * x[self.xcomp] + self.d)
        else:
            value[:] = 0

    def value_shape(self):
        return (2,)


def get_rightinv(C):
    """compute the rightinverse bmo SVD

    """
    # use numpy routine for dense matrices
    try:
        u, s, vt = np.linalg.svd(np.array(C.todense()), full_matrices=0)
    except AttributeError:
        u, s, vt = np.linalg.svd(C, full_matrices=0)

    return np.dot(vt.T, np.dot(np.diag(1.0 / s), u.T))


def get_vstar(C, ystar, odcoo, NY):

    ystarvec = get_ystarvec(ystar, odcoo, NY)
    Cgeninv = get_rightinv(C)

    return np.dot(Cgeninv, ystarvec)


def get_ystarvec(ystar, odcoo, NY):
    """get the vector of the current target signal

    """
    ymesh = dolfin.IntervalMesh(NY - 1, odcoo['ymin'], odcoo['ymax'])
    Y = dolfin.FunctionSpace(ymesh, 'CG', 1)

    ystarvec = np.zeros((NY * len(ystar), 1))
    for k, ysc in enumerate(ystar):
        cyv = dolfin.interpolate(ysc, Y)
        ystarvec[k * NY:(k + 1) * NY, 0] = cyv.vector().array()

    return ystarvec


class CharactFun(dolfin.Expression):
    """ characteristic function of subdomain """
    def __init__(self, subdom):
        self.subdom = subdom

    def eval(self, value, x):
        if self.subdom.inside(x, False):
            value[:] = 1
        else:
            value[:] = 0

    # def value_shape(self):
    #     return (2,)
