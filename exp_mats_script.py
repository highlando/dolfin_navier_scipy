from get_exp_nsmats import comp_exp_nsmats

mddir = '/afs/mpi-magdeburg.mpg.de/data/csc/projects/qbdae-nse/data/'

for N in [2, 3]:
    for Re in [1e2, 2e2, 3e2, 4e2]: 
        comp_exp_nsmats(problemname='cylinderwake', N=N, Re=Re,
                        mddir=mddir, linear_system=False)
