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

def interpolate_kdtree2d(data_in, x2, y2, elem, lons, lats, radius_of_influence=100000):
    distances, inds = create_indexes_and_distances(x2, y2, lons, lats, k=1, workers=4)
    interpolated = data_in[inds]
    interpolated[distances >= radius_of_influence] = np.nan
    interpolated[interpolated == 0] = np.nan
    
    return interpolated

    


def fint():
    parser = argparse.ArgumentParser(
        prog="pfinterp", description="Interpolates FESOM2 data to regular grid."
    )
    parser.add_argument("meshpath", help="Path to the mesh folder")
    parser.add_argument("data", help="Path to the data file")

    args = parser.parse_args()
    data = xr.open_dataset(args.data)



    x2, y2, elem = load_mesh(args.meshpath)

    left = -80
    right = -30
    bottom = 20
    top = 60
    x = np.linspace(left,right,100)
    y = np.linspace(bottom,top,100)
    lon, lat = np.meshgrid(x,y)

    distances, inds = create_indexes_and_distances(x2, y2, lon, lat, k=1, workers=4)
    data_in = data.temp[0,0,:].values

    interpolated = interpolate_kdtree2d(data_in, x2, y2, elem, lon, lat, radius_of_influence=100000)

    print(interpolated)


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    fint()