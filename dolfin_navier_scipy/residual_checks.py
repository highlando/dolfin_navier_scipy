# import numpy as np

import dolfin
from dolfin import dx, inner, nabla_grad, div, grad


def get_steady_state_res(V=None, outflowds=None, gradvsymmtrc=True, nu=None):

    def steady_state_res(vel, pres, phi=None):
        if phi is None:
            phi = dolfin.TestFunction(V)

        cnvfrm = inner(dolfin.dot(vel, nabla_grad(vel)), phi)*dx
        diffrm = nu*inner(grad(vel)+grad(vel).T, grad(phi))*dx
        if gradvsymmtrc:
            nvec = dolfin.FacetNormal(V.mesh())
            diffrm = diffrm - (nu*inner(grad(vel).T*nvec, phi))*outflowds

        pfrm = inner(pres, div(phi))*dx
        res = dolfin.assemble(diffrm+cnvfrm-pfrm)
        return res

    return steady_state_res


def get_imex_res(V=None, outflowds=None, gradvsymmtrc=True, nu=None,
                 implscheme='crni', explscheme='abtw'):
    """ define the residual for an IMEX/AB2 time discretization

    """
    if not implscheme == 'crni':
        raise NotImplementedError()

    if explscheme == 'abtw':
        def convform(cvo=None, cvt=None, phi=None):
            return (1.5*inner(dolfin.dot(cvo, nabla_grad(cvo)), phi)*dx -
                    .5*inner(dolfin.dot(cvt, nabla_grad(cvt)), phi)*dx)

    elif explscheme == 'heun':
        def convform(cvo=None, cvt=None, phi=None):
            return (.5*inner(dolfin.dot(cvo, nabla_grad(cvo)), phi)*dx +
                    .5*inner(dolfin.dot(cvt, nabla_grad(cvt)), phi)*dx)

    elif explscheme == 'eule':
        def convform(cvo=None, cvt=None, phi=None):
            return inner(dolfin.dot(cvo, nabla_grad(cvo)), phi)*dx

    def imex_res(vel, pres, dt, lastvel=None, othervel=None, phi=None):
        if phi is None:
            phi = dolfin.TestFunction(V)

        diffvel = .5*(vel+lastvel)  # Crank-Nicolson
        diffrm = nu*inner(grad(diffvel)+grad(diffvel).T, grad(phi))*dx
        if gradvsymmtrc:
            nvec = dolfin.FacetNormal(V.mesh())
            diffrm = diffrm - (nu*inner(grad(diffvel).T*nvec, phi))*outflowds
        cnvfrm = convform(cvo=lastvel, cvt=othervel, phi=phi)
        pfrm = -1./dt*inner(pres, div(phi))*dx
        dtprt = 1./dt*dolfin.assemble(inner(vel, phi)*dx) \
            - 1./dt*dolfin.assemble(inner(lastvel, phi)*dx)
        res = dolfin.assemble(diffrm+0*cnvfrm+pfrm) + dtprt
        return res

    return imex_res
