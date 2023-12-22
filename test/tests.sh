#!/bin/bash
set -o xtrace

wget https://swift.dkrz.de/v1/dkrz_c719fbc3-98ea-446c-8e01-356dac22ed90/fint/test_fint.tar
tar -xvf test_fint.tar
wget -O ./test/mesh/pi/pi_griddes_elements_IFS.nc https://gitlab.awi.de/fesom/pi/-/raw/master/pi_griddes_elements_IFS.nc
wget -O ./test/mesh/pi/pi_griddes_nodes_IFS.nc https://gitlab.awi.de/fesom/pi/-/raw/master/pi_griddes_nodes_IFS.nc
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

#cdo_remapcon
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 --interp cdo_remapcon
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-150, 150, -50, 70" --interp cdo_remapcon
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf --interp cdo_remapcon

#cdo_remaplaf
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 --interp cdo_remaplaf
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-150, 150, -50, 70" --interp cdo_remaplaf
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b arctic --interp cdo_remapcon

#smm_regrid
fint ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp smm_remapcon --no_shape_mask
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-150, 150, -50, 70" --interp smm_remaplaf
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b arctic --interp smm_remapnn
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 --interp smm_remapdis

#xesmf_regrid
fint ${FILE} ${MESH} ${INFL} -t -1 -d -1 -b "-150, 150, -50, 70" --interp xesmf_nearest_s2d
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b arctic --interp xesmf_nearest_s2d
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf --interp xesmf_nearest_s2d

#saving weights and reuse it
fint ${FILE} ${MESH} ${INFL} -t 1:5 -d -1 -b "-150, 150, -50, 70" --interp smm_remapcon --save_weights
export WEIGHTS="--weightspath ./temp.fesom.1948_interpolated_-150_150_-50_70_2.5_6125.0_1_4weighs_cdo.nc"
fint ${FILE} ${MESH} ${INFL} -t 1:5 -d -1 -b "-150, 150, -50, 70" --interp cdo_remapcon ${WEIGHTS}

fint ${FILE} ${MESH} ${INFL} -t 1:5 -d -1 -b "-150, 150, -50, 70" --interp xesmf_nearest_s2d --save_weights
export WEIGHTS="--weightspath ./temp.fesom.1948_interpolated_-150_150_-50_70_2.5_6125.0_1_4xesmf_weights.nc"
fint ${FILE} ${MESH} ${INFL} -t -1 -d -1 -b "-150, 150, -50, 70" --interp xesmf_nearest_s2d ${WEIGHTS}

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

# interpolate to a fesom (unstructured) grid
if [ ! -d "./test/mesh/core2" ]; then
    mkdir "./test/mesh/core2"
    wget -O ./test/mesh/core2/core2_old_griddes_elements_IFS.nc https://gitlab.awi.de/fesom/core2_old/-/raw/master/core2_old_griddes_elements_IFS.nc
    wget -O ./test/mesh/core2/core2_old_griddes_nodes_IFS.nc https://gitlab.awi.de/fesom/core2_old/-/raw/master/core2_old_griddes_nodes_IFS.nc
fi
export TARGET="./test/mesh/core2/"
fint ${FILE} ${MESH} ${INFL} -d -1 --target ${TARGET} --to_fesom_mesh
fint ${FILE} ${MESH} ${INFL} -t 0 -d 0:10 --target ${TARGET} --to_fesom_mesh --interp mtri_linear
fint ${FILE} ${MESH} ${INFL} -d 0:50  --target ${TARGET} --to_fesom_mesh --interp cdo_remapcon
fint ${FILE} ${MESH} ${INFL} -d 0:100  --target ${TARGET} --to_fesom_mesh --interp cdo_remaplaf

export FILE="./test/data/v.fesom.1948.nc"
fint ${FILE} ${MESH} ${INFL} -d -1 --target ${TARGET} --to_fesom_mesh
fint ${FILE} ${MESH} ${INFL} -d 0:50  --target ${TARGET} --to_fesom_mesh --interp cdo_remapcon
fint ${FILE} ${MESH} ${INFL} -d 0:100  --target ${TARGET} --to_fesom_mesh --interp cdo_remaplaf

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
