# SCLTST=500
# PRVFRM=700
# MSHPRFX='mesh/2D-double-rotcyl'

MSHDIR='/home/heiland/work/code/lqgbt-oseen/tests/mesh/2D-outlet-meshes/'
MSHNAM='karman2D-outlets'
MSHPRFX=${MSHDIR}${MSHNAM}
MSHLVL=1
RE=40
NTS=4200  # 4200 was OK with CNAB, 4000 not
TE=4
SCLTST=1.
PRVFRM=200

python3 time_dep_nse_generic.py \
    --meshprefix ${MSHPRFX} --meshlevel ${MSHLVL} \
    --Re ${RE} --Nts ${NTS} --tE ${TE} --scaletest ${SCLTST} \
    --paraviewframes ${PRVFRM}
