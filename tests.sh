#!/bin/bash
set -o xtrace

export FILE="./test/data/temp.fesom.1948.nc"
export MESH="./test/mesh/pi/"
export INFL="--influence 500000"

# minimum example
python fint.py ${FILE} ${MESH} ${INFL}

# time
python fint.py ${FILE} ${MESH} ${INFL} -t 0
python fint.py ${FILE} ${MESH} ${INFL} -t 0:10
python fint.py ${FILE} ${MESH} ${INFL} -t -1

# depths
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d -1
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0:10
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0,10,20,50
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gs

# regions
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b trop
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b arctic
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-80, -60, 20, 40" --map_projection mer
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-180, 180, 60, 90" --map_projection np
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-80, -60, 20, 40" --map_projection mer -r 100 200

# mtri_linear
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0  --interp mtri_linear
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b "-150, 150, -50, 70" --interp mtri_linear
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0 -b gulf --interp mtri_linear

# create mask
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear -o mask.nc
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear --mask mask.nc 
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp nn --mask mask.nc 
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --mask mask.nc 

# interpolate to target grid
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --target mask.nc 

# Don't apply shapely mask
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 500:3000  --interp nn --target mask.nc --no_shape_mask

# Different variables
FILE="./test/data/u.fesom.1948.nc"
python fint.py ${FILE} ${MESH} ${INFL} -t 0:3 -d -1  
# --rotate will rotate vector output and produce 2 files
python fint.py ${FILE} ${MESH} ${INFL} -t 0:3 -d 0:1000 --rotate

# 2d nodes
FILE="./test/data/ssh.fesom.1948.nc"
python fint.py ${FILE} ${MESH} ${INFL} -t 0:3 -d 100:200  

# 3d elements, on interfaces
FILE="./test/data/Av.fesom.1948.nc"
python fint.py ${FILE} ${MESH} ${INFL} -t 0:3 -d -1  

# 3d nodes, on interfaces
FILE="./test/data/Kv.fesom.1948.nc"
python fint.py ${FILE} ${MESH} ${INFL} -t 0:3 -d -1  
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d -1  --interp mtri_linear

# 2d elements
FILE="./test/data/tx_sur.fesom.1948.nc"
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0:20   
python fint.py ${FILE} ${MESH} ${INFL} -t 0 -d 0:20 --rotate

rm *.nc
