import numpy as np
import scipy.io
import json
import time

__all__ = ['output_paraview',
           'save_npa', 'save_spa',
           'load_npa', 'load_spa',
           'save_output_json',
           'plot_outp_sig',
           'load_or_comp']


def output_paraview(V=None, Q=None, VS=None, fstring='nn',
                    invinds=None, diribcs=None,
                    dbcinds=None, dbcvals=None,
                    vp=None, vc=None, pc=None, sc=None,
                    sname='nn',
                    ppin=None, t=None, tfilter=None, writeoutput=True,
                    vfile=None, pfile=None, sfile=None):
    """write the paraview output for a solution `(v,p)` or a scalar `s`

    given as coefficients
    """

    if not writeoutput:
        return
    else:
        if tfilter is not None:
            if tfilter == []:
                return
            if not t == tfilter[0]:
                return
            else:
                tfilter.pop(0)

        # hiding this calls to dolfin as much as possible
        from .dolfin_to_sparrays import expand_vp_dolfunc
        import dolfin

    if vc is not None or pc is not None or vp is not None:
        v, p = expand_vp_dolfunc(V=V, Q=Q, vp=vp,
                                 vc=vc, pc=pc,
                                 dbcinds=dbcinds, dbcvals=dbcvals,
                                 invinds=invinds, ppin=ppin,
                                 diribcs=diribcs)

        v.rename('v', 'velocity')
        # import matplotlib.pyplot as plt
        # dolfin.plot(v)
        # plt.show()
        # dolfin.plot(p)
        # plt.show()
        if vfile is None:
            vfile = dolfin.File(fstring+'_vel.pvd')
        vfile << v, t
        if p is not None:
            p.rename('p', 'pressure')
            if pfile is None:
                pfile = dolfin.File(fstring+'_p.pvd')
            pfile << p, t

    if sc is not None:
        sfun, _ = expand_vp_dolfunc(V=VS, vc=sc,
                                    dbcinds=dbcinds, dbcvals=dbcvals,
                                    invinds=invinds, diribcs=diribcs)
        sfun.rename('s', sname)

        if sfile is None:
            sfile = dolfin.File(fstring + '_' + sname + '.pvd')
        sfile << sfun, t


def save_npa(v, fstring='notspecified'):
    if fstring is None:
        print("No `fstring`, won't save anything")
        return
    np.save(fstring, v)
    return


def load_npa(fstring):
    if fstring is None:
        print("No `fstring`, won't load anything")
        return
    if not fstring[-4:] == '.npy':
        return np.load(fstring+'.npy')
    else:
        return np.load(fstring)


def save_spa(sparray, fstring):
    if fstring is None:
        print("No `fstring`, won't save anything")
        return
    scipy.io.mmwrite(fstring, sparray)


def load_spa(fstring):
    if fstring is None:
        print("No `fstring`, won't load anything")
        return
    return scipy.io.mmread(fstring).tocsc()


def load_json_dicts(StrToJs):
    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    fjs.close()
    return JsDict


def plot_prs_outp(str_to_json=None, tmeshkey='tmesh', sigkey='outsig',
                  outsig=None, tmesh=None, fignum=222, reference=None,
                  compress=5, tikzfile=None, tikzonly=False):
    import matplotlib.pyplot as plt

    if str_to_json is not None:
        jsdict = load_json_dicts(str_to_json)
        tmesh = jsdict[tmeshkey]
        outsig = jsdict[sigkey]
    else:
        str_to_json = 'notspecified'

    redinds = list(range(1, len(tmesh), compress))

    redina = np.array(redinds)

    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina],
             color='r', linewidth=2.0)

    if tikzfile is not None:
        try:
            import tikzplotlib
            tikzplotlib.save(tikzfile + '.tikz')
            print('tikz saved to ' + tikzfile + '.tikz')
        except ImportError:
            print('cannot save to tikz -- no tikzplotlib found')
    if tikzonly:
        return
    else:
        fig.show()
        return


def plot_outp_sig(str_to_json=None, tmeshkey='tmesh', sigkey='outsig',
                  outsig=None, tmesh=None, fignum=222, reference=None,
                  tikzstr=None, compress=5, tikzplease=False):
    import matplotlib.pyplot as plt

    if str_to_json is not None:
        jsdict = load_json_dicts(str_to_json)
        tmesh = jsdict[tmeshkey]
        outsig = jsdict[sigkey]
    else:
        str_to_json = 'notspecified'

    redinds = list(range(1, len(tmesh), compress))
    redina = np.array(redinds)

    NY = np.int(len(outsig[0])/2)

    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina, :NY],
             color='b', linewidth=2.0)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina, NY:],
             color='r', linewidth=2.0)
    if not tikzplease:
        plt.show()
        return

    try:
        import tikzplotlib
        if tikzstr is None:
            tikzstr = str_to_json + '{0}'.format(fignum)
        tikzplotlib.save(tikzstr + '.tikz')
        print('tikz saved to ' + tikzstr + '.tikz')
        haztikz = True
    except ImportError:
        haztikz = False
        print('cannot save to tikz -- no tikzplotlib found')

    fig.show()

    if reference is not None:
        fig = plt.figure(fignum+1)
        ax1 = fig.add_subplot(111)
        ax1.plot(tmesh, np.array(outsig)-reference)

        if haztikz:
            tikzplotlib.save(str_to_json + '{0}'.format(fignum) +
                             '_difftoref.tikz')
        fig.show()


def save_output_json(datadict=None,
                     fstring='unspecified_outputfile',
                     module='dolfin_navier_scipy.data_output_utils',
                     plotroutine='plot_outp_sig'):
    """save output to json for postprocessing

    """

    jsfile = open(fstring, mode='w')
    jsfile.write(json.dumps(datadict))

    print('output saved to ' + fstring)
    print('\n to plot run the commands \n')
    print('from ' + module + ' import ' + plotroutine)
    print(plotroutine + '("' + fstring + '")')

    if plotroutine == 'plot_outp_sig':
        print('\n to plot w/o tikz: \n')
        print('from ' + module + ' import ' + plotroutine)
        print(plotroutine + '("' + fstring + '")')


def extract_output(dictofpaths=None, tmesh=None, c_mat=None, ystarvec=None):

    cur_v = load_npa(dictofpaths[tmesh[0]])
    yn = c_mat*cur_v
    yscomplist = [yn.flatten().tolist()]
    for t in tmesh[1:]:
        cur_v = load_npa(dictofpaths[t])
        yn = c_mat*cur_v
        yscomplist.append(yn.flatten().tolist())
    if ystarvec is not None:
        ystarlist = [ystarvec(0).flatten().tolist()]
        for t in tmesh[1:]:
            ystarlist.append(ystarvec(t).flatten().tolist())

        return yscomplist, ystarlist

    else:
        return yscomplist


def meas_output_diff(dictofpaths=None, ylist=None, tmesh=None,
                     c_mat=None, ystar=None):
    if ystar is None:
        zeroy = np.zeros((c_mat.shape[0], ))

        def ystarfunc(t):
            return zeroy
    else:
        try:
            ystarfunc = ystar(0)
            ystarfunc = ystar
        except TypeError:
            def ystarfunc(t):
                return ystar.flatten()

    if ylist is None:
        ylist, ystlist = extract_output(dictofpaths=dictofpaths, tmesh=tmesh,
                                        c_mat=c_mat, ystarvec=ystarfunc)
    else:
        ystlist = [ystarfunc(t) for t in tmesh]

    try:
        diffy = np.array(ylist) - np.array(ystlist)
        ndiffy = []
        for row in range(diffy.shape[0]):
            crow = diffy[row, :]
            ndiffy.append(np.dot(crow.T, crow))
        ndiffy = np.array(ndiffy)
        dtvec = tmesh[1:] - tmesh[:-1]
        trapv = 0.5*(ndiffy[:-1] + ndiffy[1:])
        return (dtvec*trapv).sum()
    except ValueError:
        print('data does not match')
        return


def load_or_comp(filestr=None, comprtn=None, comprtnargs={},
                 arraytype=None, debug=False,
                 loadrtn=None, loadmsg='loaded ',
                 savertn=None, savemsg='saved ',
                 itsadict=False,
                 numthings=1):
    """ routine for caching computation results on disc

    Parameters
    ----------
    filestr: {string, list of strings, `None`}
        where to load/store the computed things, \
        if `None` nothing is loaded or stored
    arraytype: {`None`, 'sparse', 'dense'}
        if not None, then it sets the default routines to save/load dense or \
        sparse arrays
    itsadict: boolean, optional
        whether it is *python dictionary* that can be JSON serialized, \
        overrides all other options concerning arrays
    savertn: fun(), optional
        routine for saving the computed results, defaults to None, i.e. \
        no saving here
    debug: boolean, optional
        no saving or loading, defaults to `False`

    """
    if filestr is None or debug:
        print("no datastr specified or `debug` -- won't load/save any data")
        things = comprtn(**comprtnargs)
        return things

    if not filestr.__class__ == list:
        filestr = [filestr]

    if itsadict:
        import json
        things = []
        try:
            if debug:
                raise IOError()
            for filename in filestr:
                fjs = open(filename)
                things.append(json.load(fjs))
                fjs.close()
        except IOError:
            things = comprtn(**comprtnargs)
            if things.__class__ == dict:
                f = open(filestr[0], 'w')
                f.write(json.dumps(things))
                f.close()
            else:
                for k, filename in enumerate(filestr):
                    f = open(filename, 'w')
                    f.write(json.dumps(things[k]))
                    f.close()
        if len(things) == 1:
            return things[0]
        else:
            return things

    if arraytype == 'dense':
        savertn = save_npa
        loadrtn = load_npa
    elif arraytype == 'dense':
        savertn = save_spa
        loadrtn = load_spa

    if numthings == 1:
        filestr = filestr[0]  # TODO: make this right (for multiple outputs)
        try:
            thing = loadrtn(filestr)
            print(loadmsg + filestr)
        except IOError:
            print('could not load ' + filestr + ' -- lets compute it')
            thing = comprtn(**comprtnargs)
            if savertn is not None:
                savertn(thing, filestr)
                print(savemsg + filestr)
        return thing
    if numthings == 2:
        try:
            thing1 = loadrtn(filestr[0])
            thing2 = loadrtn(filestr[1])
            print(loadmsg + filestr[0] + '/' + filestr[1])
        except IOError:
            print('could not load ' + filestr[0] + ' -- lets compute it')
            print('could not load ' + filestr[1] + ' -- lets compute it')
            thing1, thing2 = comprtn(**comprtnargs)
            if savertn is not None:
                savertn(thing1, filestr[0])
                savertn(thing2, filestr[1])
                print(savemsg + filestr[0] + '/' + filestr[1])
        return thing1, thing2


def logtofile(logstr):
    import datetime
    import sys

    print('log gogoes ' + logstr)
    print('how about \ntail -f '+logstr)
    sys.stdout = open(logstr, 'a', 1)
    print(('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).
          format('X', str(datetime.datetime.now())))
    # from contextlib import redirect_stdout
    # with open(logstr, 'w') as f:
    #     with redirect_stdout(f):
    #         print(('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).
    #               format('X', str(datetime.datetime.now())))


class Timer(object):
    def __init__(self, name=None, logger=None, timerinfo={}, verbose=True):
        self.name = name
        self.logger = logger
        self.timerinfo = timerinfo
        self.verbose = verbose

    def __enter__(self):
        self.tstart = time.time()
        # self.ptstart = time.process_time()

    def __exit__(self, type, value, traceback):
        elt = time.time() - self.tstart
        # elpt = time.process_time() - self.ptstart
        self.timerinfo.update(dict(elt=elt))
        if self.logger is not None:
            self.logger.info('{0}: Elapsed time: {1}'.
                             format(self.name, elt))
        if self.verbose:
            print('{0}: Elapsed time: {1}'.format(self.name, elt))
            # print('{0}: Elapsed CPU time: {1}'.format(self.name, elpt))
