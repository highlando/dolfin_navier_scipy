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
