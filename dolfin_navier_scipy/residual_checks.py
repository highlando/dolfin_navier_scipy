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


def get_cnabtwo_res(V=None, outflowds=None, gradvsymmtrc=True, nu=None):
    """ define the residual for an IMEX/AB2 time discretization

    """

    def cnabtwo_res(vel, pres, lastvel, lastlastvel, dt, phi=None):
        if phi is None:
            phi = dolfin.TestFunction(V)

        cnvfrm = 1.5*inner(dolfin.dot(lastvel, nabla_grad(lastvel)), phi)*dx -\
            .5*inner(dolfin.dot(lastlastvel, nabla_grad(lastlastvel)), phi)*dx
        diffrm = nu*inner(grad(vel)+grad(vel).T, grad(phi))*dx
        if gradvsymmtrc:
            nvec = dolfin.FacetNormal(V.mesh())
            diffrm = diffrm - (nu*inner(grad(vel).T*nvec, phi))*outflowds
        pfrm = inner(pres, div(phi))*dx
        dtprt = dolfin.assemble(inner(1./dt*vel, phi))*dx \
            - dolfin.assemble(inner(1./dt*lastvel, phi))*dx
        res = dolfin.assemble(diffrm+cnvfrm-pfrm) + dtprt
        return res

    return cnabtwo_res
