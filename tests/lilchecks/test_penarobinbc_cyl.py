import dolfin

import dolfin_navier_scipy.dolfin_to_sparrays as dts
# import dolfin_navier_scipy.stokes_navier_utils as snu
import dolfin_navier_scipy.problem_setups as dnsps

import matplotlib.pyplot as plt


def check_penaro(problemname='cylinderwake', N=2,
                 Re=1.0e2, t0=0.0, tE=2.0, Nts=1e3+1, plot=False):

    femp = dnsps.cyl_fems(N, bccontrol=True, verbose=True)
    V = femp['V']

    nu = femp['charlen']/Re

    bcsd = femp['contrbcssubdomains']
    bcssfuns = femp['contrbcsshapefuns']

    stokesmats = dts.get_stokessysmats(V, femp['Q'], nu=nu,
                                       bccontrol=True, cbclist=bcsd,
                                       cbshapefuns=bcssfuns)
    if plot:
        plt.figure(2)
        plt.spy(stokesmats['amatrob'])

    print('Number of nonzeros in amatrob:', stokesmats['amatrob'].nnz)
    plt.show()


def check_ass_penaro(bcsd=None, bcssfuns=None, V=None, plot=False):
    mesh = V.mesh()

    bcone = bcsd[1]
    contshfunone = bcssfuns[1]
    Gammaone = bcone()

    bparts = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    Gammaone.mark(bparts, 0)

    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    # Robin boundary form
    arob = dolfin.inner(u, v) * dolfin.ds(0)
    brob = dolfin.inner(v, contshfunone) * dolfin.ds(0)

    amatrob = dolfin.assemble(arob, exterior_facet_domains=bparts)
    bmatrob = dolfin.assemble(brob, exterior_facet_domains=bparts)

    amatrob = dts.mat_dolfin2sparse(amatrob)
    amatrob.eliminate_zeros()
    print('Number of nonzeros in amatrob:', amatrob.nnz)
    bmatrob = bmatrob.array()  # [ININDS]

    if plot:
        plt.figure(2)
        plt.spy(amatrob)

    if plot:
        plt.figure(1)
        for x in contshfunone.xs:
            plt.plot(x[0], x[1], 'bo')

    plt.show()


def checktheboundarycoordinates(bcsd, femp, plot=False):
    g1 = dolfin.Constant((0, 0))
    for bc in bcsd:
        bcrl = dolfin.DirichletBC(femp['V'], g1, bc())
        bcdict = bcrl.get_boundary_values()
        print(list(bcdict.keys()))

    bcinds = list(bcdict.keys())

    V = femp['V']

    cylmesh = femp['V'].mesh()
    if plot:
        dolfin.plot(cylmesh)
        dolfin.interactive(True)

    gdim = cylmesh.geometry().dim()
    dofmap = V.dofmap()

    # Get coordinates as len(dofs) x gdim array
    dofs_x = dofmap.tabulate_all_coordinates(cylmesh).reshape((-1, gdim))

    # for dof, dof_x in zip(dofs, dofs_x):
    #     print dof, ':', dof_x
    xcenter = 0.2
    ycenter = 0.2
    for bcind in bcinds:
        dofx = dofs_x[bcind, :]
        dx = dofx[0] - xcenter
        dy = dofx[1] - ycenter
        r = dolfin.sqrt(dx*dx + dy*dy)
        print(bcind, ':', dofx, r)

if __name__ == '__main__':
    check_penaro(plot=True, N=2)
