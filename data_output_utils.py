import numpy as np
import scipy.io
from dolfin_to_sparrays import expand_vp_dolfunc
import dolfin


def output_paraview(V=None, Q=None, fstring='nn',
                    invinds=None, diribcs=None,
                    vp=None, t=None, writeoutput=True):
    """write the paraview output for a solution vector vp

    """

    if not writeoutput:
        return

    v, p = expand_vp_dolfunc(V=V, Q=Q, vp=vp,
                             invinds=invinds,
                             diribcs=diribcs)

    v.rename('v', 'velocity')
    p.rename('p', 'pressure')

    pfile = dolfin.File(fstring+'_p.pvd')
    vfile = dolfin.File(fstring+'_vel.pvd')

    vfile << v, t
    pfile << p, t


def save_npa(v, fstring='notspecified'):
    np.save(fstring, v)
    return


def load_npa(fstring):
    return np.load(fstring+'.npy')


def save_spa(sparray, fstring='notspecified'):
    scipy.io.mmwrite(fstring, sparray)


def load_spa(fstring):
    return scipy.io.mmread(fstring).tocsc()
