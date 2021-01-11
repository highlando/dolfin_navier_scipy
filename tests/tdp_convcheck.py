import numpy as np
import matplotlib.pyplot as plt

import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps


def conv_plot(abscissa, datalist, fit=None,
              markerl=None, xlabel=None, ylabel=None,
              fititem=0, fitfac=1.,
              leglist=None, legloc=1,
              title='title not provided', fignum=None,
              ylims=None, xlims=None,
              yticks=None,
              logscale=False, logbase=10,
              tikzfile=None, showplot=True):
    """Universal function for convergence plots

    Parameters
    ----------
    fititem : integer, optional
        to which item of the data the fit is aligned, defaults to `0`
    fitfac : float, optional
        to shift the fitting lines in y-direction, defaults to `1.0`
    """

    try:
        import seaborn as sns
        sns.set(style="whitegrid")
        # mpilightgreen = '#BFDFDE'
        # mpigraygreen = '#7DA9A8'
        # sns.set_palette(sns.dark_palette(mpigraygreen, 4, reverse=True))
        # sns.set_palette(sns.dark_palette(mpilightgreen, 6, reverse=True))
        # sns.set_palette('cool', 3)
        sns.set_palette('ocean_r', 7)
    except ImportError:
        print('I recommend to install seaborn for nicer plots')

    lend = len(datalist)
    if markerl is None:
        markerl = ['']*lend
    if leglist is None:
        leglist = [None]*lend

    plt.figure(fignum)
    ax = plt.axes()

    for k, data in enumerate(datalist):
        plt.plot(abscissa, data, markerl[k], label=leglist[k])

    if fit is not None:
        fls = [':', ':']
        for i, cfit in enumerate(fit):
            abspow = []
            for ela in abscissa:
                try:
                    abspow.append((ela/abscissa[0])**(-cfit) *
                                  datalist[0][fititem]*fitfac)
                except TypeError:
                    abspow.append((ela/abscissa[0])**(-cfit) *
                                  datalist[0][0][fititem]*fitfac)
            ax.plot(abscissa, abspow, 'k'+fls[i])

    if logscale:
        ax.set_xscale('log', base=logbase)
        ax.set_yscale('log', base=logbase)
    if ylims is not None:
        plt.ylim(ylims)
    if xlims is not None:
        plt.xlim(xlims)
    if yticks is not None:
        plt.yticks(yticks)
    if title is not None:
        ax.set_title(title)

    plt.legend(loc=legloc)
    plt.grid(which='major')
    # _gohome(tikzfile, showplot)
    plt.show()
    return


def cnvchk(meshprfx='mesh/karman2D-outlets', meshlevel=1, proutdir='results/',
           problem='drivencavity', N=None, nu=1e-2, Re=None,
           time_int_scheme='cnab',
           t0=0.0, tE=1.0, Nts=1e2+1, scheme='TH', dblng=2):

    meshfile = meshprfx + '_lvl{0}.xml.gz'.format(meshlevel)
    physregs = meshprfx + '_lvl{0}_facet_region.xml.gz'.format(meshlevel)
    geodata = meshprfx + '_geo_cntrlbc.json'

    femp, stokesmatsc, rhsd = \
        dnsps.get_sysmats(problem='gen_bccont', Re=Re, bccontrol=False,
                          scheme=scheme, mergerhs=True,
                          meshparams=dict(strtomeshfile=meshfile,
                                          strtophysicalregions=physregs,
                                          strtobcsobs=geodata))
    # setting some parameters
    if Re is not None:
        nu = femp['charlen']/Re
    soldict = stokesmatsc  # containing A, J, JT
    soldict.update(femp)  # adding V, Q, invinds, diribcs

    soldict.update(fv=rhsd['fv'], fp=rhsd['fp'],
                   N=N, nu=nu, return_final_vp=True,
                   start_ssstokes=True,
                   get_datastring=None,
                   verbose=True,
                   treat_nonl_explicit=True,
                   time_int_scheme=time_int_scheme,
                   dbcinds=femp['dbcinds'], dbcvals=femp['dbcvals'])

    mmat = stokesmatsc['M']

    cnts = Nts*2**dblng
    tips = dict(t0=t0, tE=tE, Nts=cnts)
    soldict.update(tips)  # adding time integration params
    print('*** computing the ref solution ***')
    (vfref, pfref) = snu.solve_nse(**soldict)
    print('*** done with the ref solution ***')

    soldict.update(dict(verbose=False))

    errlst, ntslst = [], []
    for k in range(dblng):
        cnts = Nts*2**k
        tips = dict(t0=t0, tE=tE, Nts=cnts)
        soldict.update(tips)  # adding time integration params
        (vf, pf) = snu.solve_nse(**soldict)
        difv = vf - vfref
        cnv = np.sqrt(difv.T @ mmat @ difv).flatten()[0]
        errlst.append(cnv)
        ntslst.append(cnts)
        print('Nts: {0} -- |v-vref|: {1:e}'.format(cnts, cnv))

    conv_plot(ntslst, [errlst], logscale=True, fit=[2], markerl=['o'],
              leglist=[time_int_scheme],
              title='Check for 2nd order convergence')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshprefix", type=str,
                        help="prefix for the mesh files",
                        default='mesh/karman2D-outlets')
    parser.add_argument("--meshlevel", type=int,
                        help="mesh level", default=1)
    parser.add_argument("--Re", type=int,
                        help="Reynoldsnumber", default=100)
    parser.add_argument("--tE", type=float,
                        help="final time of the simulation", default=.1)
    parser.add_argument("--Nts", type=float,
                        help="number of time steps", default=100)
    parser.add_argument("--scaletest", type=float,
                        help="scale the test size", default=1.)
    parser.add_argument("--paraviewframes", type=int,
                        help="number of outputs for paraview", default=200)
    parser.add_argument("--doublings", type=int,
                        help="how often we double the time steps", default=4)
    parser.add_argument("--tis", type=str, choices=['cnab', 'sbdf2'],
                        help="scheme for time integration", default='sbdf2')
    args = parser.parse_args()
    print(args)
    scheme = 'TH'

    cnvchk(problem='gen_bccont', Re=args.Re,
           meshprfx=args.meshprefix, meshlevel=args.meshlevel,
           t0=0., tE=args.scaletest*args.tE,
           Nts=np.int(args.scaletest*args.Nts),
           dblng=args.doublings,
           scheme=scheme, time_int_scheme=args.tis)
