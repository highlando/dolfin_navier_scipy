MSHPRFX='/home/heiland/work/code/lqgbt-oseen/tests/mesh/2D-outlet-meshes/karman2D-outlets'
python3 steadystate_generic.py \
    --meshprefix ${MSHPRFX} --meshlevel 2 \
    --Re 40
