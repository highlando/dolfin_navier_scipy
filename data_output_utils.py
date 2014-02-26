import numpy as np
import scipy.io
import json
from dolfin_to_sparrays import expand_vp_dolfunc
import dolfin


def output_paraview(V=None, Q=None, fstring='nn',
                    invinds=None, diribcs=None,
                    vp=None, vc=None, pc=None,
                    t=None, writeoutput=True,
                    vfile=None, pfile=None):
    """write the paraview output for a solution vector vp

    """

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


def load_json_dicts(StrToJs):
    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    return JsDict


def save_output_json(datadict=None,
                     fstring='unspecified_outputfile',
                     importcall='from module import plotroutine',
                     plotroutine='plotroutine'):
    """save output to json for postprocessing

    """

    jsfile = open(fstring, mode='w')
    jsfile.write(json.dumps(datadict))

    print 'output saved to ' + fstring
    print '\n to plot run the commands \n'
    print importcall
    print plotroutine + '("' + fstring + '")'


def extract_output(dictofpaths=None, tmesh=None, c_mat=None, ystarvec=None):

    cur_v = load_npa(dictofpaths[tmesh[0]])
    yn = c_mat*cur_v
    yscomplist = [yn.flatten().tolist()]
    for t in tmesh[1:]:
        cur_v = load_npa(dictofpaths[tmesh[t]])
        yn = c_mat*cur_v
        yscomplist.append(yn.flatten().tolist())
    if ystarvec is not None:
        ystarlist = [ystarvec(0).flatten().tolist()]
        for t in tmesh[1:]:
            ystarlist.append(ystarvec(0).flatten().tolist())

        return yscomplist, ystarlist

    else:
        return yscomplist
