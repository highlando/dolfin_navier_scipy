import numpy as np
import scipy.io
from dolfin_to_nparrays import expand_vp_dolfunc


def output_paraview(tip, femp, vp=None, t=None):
    """write the paraview output for a solution vector vp

    """

    v, p = expand_vp_dolfunc(V=femp['V'], Q=femp['Q'], vp=vp,
                             invinds=femp['invinds'],
                             diribcs=femp['diribcs'])

    v.rename('v', 'velocity')
    p.rename('p', 'pressure')

    tip['vfile'] << v, t
    tip['pfile'] << p, t


def save_npa(v, fstring='notspecified'):
    np.save(fstring, v)
    return


def load_npa(fstring):
    return np.load(fstring+'.npy')


def save_spa(sparray, fstring='notspecified'):
    scipy.io.mmwrite(fstring, sparray)


def load_spa(fstring):
    return scipy.io.mmread(fstring).tocsc()
