# SCLTST=500
# PRVFRM=700
# MSHPRFX='mesh/2D-double-rotcyl'

MSHNM='/karman2D-outlets'
MSHDR='/home/heiland/work/code/lqgbt-oseen/tests/mesh/2D-outlet-meshes'
MSHPRFX=${MSHDR}${MSHNM}
MSHLVL=2
RE=50
NTS=100  # 4200 was OK with CNAB, 4000 not
TE=.1
SCLTST=1.
DBLNGS=5
TIS='cnab'
TIS='sbdf2'

python3 tdp_convcheck.py \
    --meshprefix ${MSHPRFX} --meshlevel ${MSHLVL} \
    --doublings ${DBLNGS} --tis ${TIS} \
    --Re ${RE} --Nts ${NTS} --tE ${TE} --scaletest ${SCLTST}
