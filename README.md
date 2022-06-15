# fint
Easy interpolation of [fesom2](https://github.com/FESOM/fesom2) output data. Your data should be created with `refactoring` brunch of the model.

## Motivation

Make something that is more light weight to install compared to [pyfesom2](https://github.com/FESOM/pyfesom2) that has similar capabilities, but sometimes could be pain to install.

## Installation

On most HPC machines you should be able to select python module, that have `xarray` and preferably `cartopy` installed, clone repository, and just use it simply as a script (e.g. `python /path/to/fint/fint/fint.py [some options]`). But if you want to install it in a bit more cleaner way, read instructions below.

### Dependencies

- xarray
- netcdf4
- numpy
- scipy
- matplotlib
- pandas
#### Optional
- cartopy (should work without it as well, but ony interpolate to lan lat boxes)
- shapely (also should work without it)

## Miniconda installation

Please follow [those instructions](https://github.com/koldunovn/python_for_geosciences#getting-started-for-linuxmac) to install Miniconda.

Change to `fint` directory and create separate `fint` environment with:

```shell
conda env create -f environment.yml
```

activate it after installation with

```shell
conda activate fint
```

and you should be good to go :)


## Basic functionality

The easiest way to test `fint` is to execute `tests.sh` script, that will download sample data and perform some tests. Let's assume you executed the script once, and everything worked. We now can repeat some of the commands from there with explanations.

We set some variables, that will be frequently used as command line arguments:
```
export FILE="./test/data/temp.fesom.1948.nc"
export MESH="./test/mesh/pi/"
export INFL="--influence 500000"
```

As bare minimum you should provide path to FESOM2 output file and path to the mesh. We also use `--influence`, so that output result looks a bit nicer:
```shell
python fint.py ${FILE} ${MESH} ${INFL}
```

