# import numpy as np

import dolfin
from dolfin import dx, inner, nabla_grad, div, grad

import dolfin_navier_scipy.dolfin_to_sparrays as dts
from sadptprj_riclyap_adi.lin_alg_utils import app_prj_via_sadpnt


__all__ = ['prjctd_steadystate_res',
           'get_steady_state_res',
           'get_imex_res',
           ]


def prjctd_steadystate_res(vvec=None, mmat=None, amat=None, jmat=None, fv=None,
                           invinds=None, dbcvals=None, dbcinds=None,
                           stokes_only=False, V=None):
    ''' compute 

    Pi.T ( A*v + N(v) - fv )


    where Pi = I - M.-1 J.T S.-1 J

    ''' 
    
    if stokes_only:
        fres = amat@vvec - fv
    else:
        cnvec = dts.get_convvec(u0_vec=vvec, V=V, uone_utwo_same=True,
                                invinds=invinds,
                                dbcinds=dbcinds, dbcvals=dbcvals)
        fres = amat@vvec + cnvec - fv
    prjres = app_prj_via_sadpnt(amat=mmat, jmat=jmat, rhsv=fres,
                                transposedprj=True)

    return prjres

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

        pfrm = -inner(pres, div(phi))*dx
        dtprt = 1./dt*dolfin.assemble(inner(vel, phi)*dx) \
            - 1./dt*dolfin.assemble(inner(lastvel, phi)*dx)
        res = dolfin.assemble(diffrm+cnvfrm+pfrm) + dtprt

        # import numpy as np
        # nfc_c = dolfin.assemble(cnvfrm).get_local()
        # print(np.linalg.norm(nfc_c), nfc_c[0], nfc_c.size)
        # print('debggng')
        return res

    return imex_res
