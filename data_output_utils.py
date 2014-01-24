import numpy as np
import scipy.io
from dolfin_to_sparrays import expand_vp_dolfunc
import dolfin


def output_paraview(V=None, Q=None, fstring='nn',
                    invinds=None, diribcs=None,
                    vp=None, vc=None, pc=None,
                    t=None, writeoutput=True,
                    vfile=None, pfile=None):
    """write the paraview output for a solution vector vp

    """
    print t

    if not writeoutput:
        return

    v, p = expand_vp_dolfunc(V=V, Q=Q, vp=vp,
                             vc=vc, pc=pc,
                             invinds=invinds,
                             diribcs=diribcs)

    v.rename('v', 'velocity')
    if vfile is None:
        vfile = dolfin.File(fstring+'_vel.pvd')
    vfile << v, t
    if p is not None:
        p.rename('p', 'pressure')
        if pfile is None:
            pfile = dolfin.File(fstring+'_p.pvd')
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
