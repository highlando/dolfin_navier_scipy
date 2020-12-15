import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps


def nsestst(meshprfx='mesh/karman2D-outlets', meshlevel=1, proutdir='results/',
            problem='drivencavity', N=None, nu=1e-2, Re=None,
            ParaviewOutput=False, scheme='TH', bccontrol=False, palpha=1e5):

    meshfile = meshprfx + '_lvl{0}.xml.gz'.format(meshlevel)
    physregs = meshprfx + '_lvl{0}_facet_region.xml.gz'.format(meshlevel)
    geodata = meshprfx + '_geo_cntrlbc.json'
    meshparams = dict(strtomeshfile=meshfile,
                      strtophysicalregions=physregs,
                      strtobcsobs=geodata)

    initssres = [40, 60, 80]
    v_init = None
    for initre in initssres:
        initsssoldict = {}
        if initre >= Re:
            initre = Re
            initsssoldict.update(paraviewoutput=ParaviewOutput,
                                 vfileprfx=proutdir+'vel_',
                                 pfileprfx=proutdir+'p_')
        else:
            print('Initialising solution with Re={0}'.format(initre))

        initssfemp, initssstokesmatsc, initssrhsd = \
            dnsps.get_sysmats(problem='gen_bccont', Re=initre,
                              bccontrol=bccontrol, scheme='TH', mergerhs=True,
                              meshparams=meshparams)

        if bccontrol:
            initssstokesmatsc['A'] = initssstokesmatsc['A'] \
                + 1./palpha*initssstokesmatsc['Arob']

        initsssoldict.update(initssstokesmatsc)
        initsssoldict.update(initssfemp)
        initssnu = initssfemp['charlen']/initre
        initsssoldict.update(fv=initssrhsd['fv'], fp=initssrhsd['fp'],
                             nu=initssnu)
        vp_ss_nse = snu.\
            solve_steadystate_nse(vel_pcrd_stps=15, return_vp=True,
                                  vel_start_nwtn=v_init,
                                  vel_nwtn_tol=4e-13,
                                  **initsssoldict)

        v_init = vp_ss_nse[0]
        if initre == Re:
            break

    print('for plots check \nparaview ' + proutdir + 'vel___steadystates.pvd')
    print('or \nparaview ' + proutdir + 'p___steadystates.pvd')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshprefix", type=str,
                        help="prefix for the mesh files",
                        default='mesh/karman2D-outlets')
    parser.add_argument("--meshlevel", type=int,
                        help="mesh level", default=1)
    parser.add_argument("--Re", type=int,
                        help="Reynoldsnumber", default=60)
    args = parser.parse_args()
    print(args)
    scheme = 'TH'

    nsestst(problem='gen_bccont', Re=args.Re,
            meshprfx=args.meshprefix, meshlevel=args.meshlevel,
            scheme=scheme, ParaviewOutput=True)
