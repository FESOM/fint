import xarray as xr
import cartopy
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, NearestNDInterpolator
import matplotlib.tri as mtri
import matplotlib.pylab as plt
import pandas as pd
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.spatial import cKDTree
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import gc
import argparse
from regions import define_region

def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    calculates lon, lat coordinates of a point on a sphere with
    radius R. Taken from http://earthpy.org/interpolation_between_grids_with_ckdtree.html
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def create_indexes_and_distances(model_lon, model_lat, lons, lats, k=1, workers=2):
    """
    Creates KDTree object and query it for indexes of points in FESOM mesh that are close to the
    points of the target grid. Also return distances of the original points to target points.
    Parameters
    ----------
    mesh : fesom_mesh object
        pyfesom mesh representation
    lons/lats : array
        2d arrays with target grid values.
    k : int
        k-th nearest neighbors to return.
    n_jobs : int, optional
        Number of jobs to schedule for parallel processing. If -1 is given
        all processors are used. Default: 1.
    Returns
    -------
    distances : array of floats
        The distances to the nearest neighbors.
    inds : ndarray of ints
        The locations of the neighbors in data.
    """
    xs, ys, zs = lon_lat_to_cartesian(model_lon, model_lat)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k, workers=workers)

    return distances, inds

def ind_for_depth(depth, depths_from_file):
    """
    Find the model depth index that is closest to the required depth
    Parameters
    ----------
    depth : float
        desired depth.
    mesh : object
        FESOM mesh object
    Returns
    dind : int
        index that corresponds to the model depth level closest to `depth`.
    -------
    """
    arr = [abs(abs(z) - abs(depth)) for z in depths_from_file]
    v, i = min((v, i) for (i, v) in enumerate(arr))
    dind = i
    return dind

def load_mesh(mesh_path):
    nodes = pd.read_csv(
            mesh_path+'/nod2d.out',
            delim_whitespace=True,
            skiprows=1,
            names=["node_number", "x", "y", "flag"],
        )

    x2 = nodes.x.values
    y2 = nodes.y.values

    file_content = pd.read_csv(
            mesh_path+'/elem2d.out',
            delim_whitespace=True,
            skiprows=1,
            names=["first_elem", "second_elem", "third_elem"],
        )

    elem = file_content.values - 1

    return x2, y2, elem

def interpolate_kdtree2d(data_in, x2, y2, elem, lons, lats, distances, inds, radius_of_influence=100000):
    
    interpolated = data_in[inds]
    interpolated[distances >= radius_of_influence] = np.nan
    interpolated[interpolated == 0] = np.nan
    interpolated.shape = lons.shape
    
    return interpolated

def parse_depths(depths, depths_from_file):
    if len(depths.split(",")) > 1:
        depths = list(map(int, depths.split(",")))
    elif int(depths) == -1:
        depths = [-1]
    else:
        depths = [int(depths)]
    # print(depths)

    if depths[0] == -1:
        dinds = range(depths_from_file.shape[0])
        realdepths = depths_from_file

    else:
        dinds = []
        realdepths = []
        for depth in depths:
            ddepth = ind_for_depth(depth, depths_from_file)
            dinds.append(ddepth)
            realdepths.append(depths_from_file[ddepth])
    # print(dinds)
    # print(realdepths)
    return dinds, realdepths

def parse_timesteps(timesteps, time_shape):

    if len(timesteps.split(":")) == 2:
        y = range(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
        # y = slice(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
    elif len(timesteps.split(":")) == 3:
        if timesteps.split(":")[1] == "end":
            stop = time_shape
        else:
            stop = int(timesteps.split(":")[1])
        y = range(int(timesteps.split(":")[0]), stop, int(timesteps.split(":")[2]))
        # y = slice(int(timesteps.split(":")[0]),
        #           int(timesteps.split(":")[1]),
        #           int(timesteps.split(":")[2]))
    elif len(timesteps.split(",")) > 1:
        y = list(map(int, timesteps.split(",")))
    elif int(timesteps) == -1:
        y = -1
    else:
        y = [int(timesteps)]
    timesteps = y
    print("timesteps {}".format(timesteps))
    return timesteps

def fint():
    parser = argparse.ArgumentParser(
        prog="pfinterp", description="Interpolates FESOM2 data to regular grid."
    )
    parser.add_argument("meshpath", help="Path to the mesh folder")
    parser.add_argument("data", help="Path to the data file")
    parser.add_argument(
        "--depths",
        "-d",
        default="0",
        type=str,
        help="Depths in meters. \
            Closest values from model levels will be taken.\
            Several options available: number - e.g. '100',\
                                       coma separated list - e.g. '0,10,100,200',\
                                       -1 - all levels will be selected.",
    )

    parser.add_argument(
        "--timesteps",
        "-t",
        default="-1",
        type=str,
        help="Explicitly define timesteps of the input fields. There are several options:\
            '-1' - all time steps, number - one time step (e.g. '5'), numbers - coma separated (e.g. '0, 3, 8, 10'), slice - e.g. '5:10',\
            slice with steps - e.g. '8:120:12'.\
            slice untill the end of the time series - e.g. '8:end:12'.",
    )

    parser.add_argument(
        "--box",
        "-b",
        # nargs=4,
        type=str,
        default="-180.0, 180.0, -80.0, 90.0",
        help="Map boundaries in -180 180 -90 90 format that will be used for interpolation.",
        # metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
    )

    parser.add_argument(
        "--res",
        "-r",
        nargs=2,
        type=int,
        # default=(360, 170),
        help="Number of points along each axis that will be used for interpolation (for lon and  lat).",
        metavar=("N_POINTS_LON", "N_POINTS_LAT"),
    )

    parser.add_argument(
        "--influence",
        "-i",
        default=80000,
        type=float,
        help="Radius of influence for interpolation, in meters.",
    )

    parser.add_argument(
        "--map_projection",
        "-m",
        # nargs=4,
        type=str,
        # default="-180.0, 180.0, -80.0, 90.0",
        help="Map boundaries in -180 180 -90 90 format that will be used for interpolation.",
        # metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
    )


    args = parser.parse_args()
    data = xr.open_dataset(args.data)
    radius_of_influence = args.influence
    projection = args.map_projection

    variable_name = list(data.data_vars)[0]
    dim_names = list(data.coords)
        
    if ("nz1" in data.dims) or ("nz" in data.dims):
        depth_coord = dim_names[0]
        depths_from_file = data[depth_coord].values
        # print(data)
        dinds, realdepths = parse_depths(args.depths, depths_from_file)
        # print(dinds)
    else:
        dinds = [0]
        realdepths = [0]

    time_shape = data.time.shape[0]
    timesteps = parse_timesteps(args.timesteps, time_shape)
    if timesteps == -1:
        timesteps = range(time_shape)
    # set timestep to 0 if data have only one time step
    if time_shape == 1:
        timesteps = [0]

    print(timesteps)

    x2, y2, elem = load_mesh(args.meshpath)

    x, y, lon, lat = define_region(args.box, args.res, projection)

    depth_limit_up = 0
    depth_limit_down = 2
    time_start = 0
    time_end = 5
    distances, inds = create_indexes_and_distances(x2, y2, lon, lat, k=1, workers=4)
    interpolated3d = np.zeros((len(timesteps), len(realdepths), len(y), len(x)))
    for t_index, ttime in enumerate(timesteps):
        for d_index, (dind, realdepth) in enumerate(zip(dinds, realdepths)):
            print(ttime)
            data_in = data[variable_name][ttime, dind, :].values
            interpolated = interpolate_kdtree2d(data_in, x2, y2, elem, lon, lat, distances, inds, radius_of_influence=radius_of_influence)
            interpolated3d[t_index, d_index, :, :] = interpolated

    out1 = xr.Dataset({variable_name:(['time', 'depth', 'lat', 'lon'], interpolated3d)},
                     coords={'time':np.atleast_1d(data.time.data[timesteps]),
                    'depth':realdepths,
                  'lon':(['lon'], x),
                  'lat':(['lat'], y )
                 })

    out1.to_netcdf(args.data.replace(".nc", "_interpolated.nc"))

    print(out1)


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    fint()


