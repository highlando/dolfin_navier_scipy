import numpy as np

Nts = 123
nsect = 10
trange = np.linspace(0, 1, Nts)
lensect = np.int(np.floor(Nts/nsect))

loctrngs = []
for k in np.arange(nsect-1):
    loctrngs.append(trange[k*lensect: (k+1)*lensect+1])
loctrngs.append(trange[(nsect-1)*lensect:])

for lctrng in loctrngs:
    print(lctrng)
