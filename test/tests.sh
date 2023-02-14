#!/bin/bash
set -o xtrace

wget https://swift.dkrz.de/v1/dkrz_c719fbc3-98ea-446c-8e01-356dac22ed90/fint/test_fint.tar
tar -xvf test_fint.tar

export FILE="./test/data/temp.fesom.1948.nc"
export MESH="./test/mesh/pi/"
export INFL="--influence 500000"

# minimum example
fint ${FILE} ${MESH} ${INFL}

# time
fint ${FILE} ${MESH} ${INFL} -t 0
fint ${FILE} ${MESH} ${INFL} -t 0:10
fint ${FILE} ${MESH} ${INFL} -t -1

# depths
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0:10
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0,10,20,50
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gs

# regions
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b trop
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b arctic
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-80, -60, 20, 40" --map_projection mer
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-180, 180, 60, 90" --map_projection np
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-80, -60, 20, 40" --map_projection mer -r 100 200

# mtri_linear
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0  --interp mtri_linear
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-150, 150, -50, 70" --interp mtri_linear
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf --interp mtri_linear

# create mask
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear -o mask.nc
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear --mask mask.nc
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp nn --mask mask.nc
fint ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --mask mask.nc
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp nn --no_mask_zero

# interpolate to target grid
fint ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --target mask.nc

# Don't apply shapely mask
fint ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --target mask.nc --no_shape_mask

# Different variables
FILE="./test/data/u.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d -1
# --rotate will rotate vector output and produce 2 files
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d 0:1000 --rotate

# 2d nodes
FILE="./test/data/ssh.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d 100:200

# 3d elements, on interfaces
FILE="./test/data/Av.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d -1
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d -1  --interp linear_scipy

# 3d nodes, on interfaces
FILE="./test/data/Kv.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0:3 -d -1
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear

# 2d elements
FILE="./test/data/tx_sur.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0:20
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0:20 --rotate

# timedelta
export FILE="./test/data/temp.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -t 0:10 --timedelta '10D'
fint ${FILE} ${MESH} ${INFL} -t 0:10 --timedelta ' -10D'
fint ${FILE} ${MESH} ${INFL} -t 0:10 --timedelta '1h'
fint ${FILE} ${MESH} ${INFL} -t 0:10 --timedelta ' -1h'

rm *.nc
