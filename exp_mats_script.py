from get_exp_nsmats import comp_exp_nsmats

# mddir = '/afs/mpi-magdeburg.mpg.de/data/csc/projects/qbdae-nse/data/'
mddir = 'data/'

relist = [x*10**y for x in range(1, 14, 2) for y in [2]]
for N in [4]:  # 10, 15, 20, 25]:  # , 1, 2]:
    for Re in relist:
        # comp_exp_nsmats(problemname='cylinderwake', N=N, Re=Re,
        comp_exp_nsmats(problemname='drivencavity', N=N, Re=Re,
                        mddir=mddir, linear_system=False)
