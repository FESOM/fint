import argparse
import os
from platform import node

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import sparse
import subprocess
from smmregrid import Regridder
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
from scipy.spatial import cKDTree

from .regions import define_region, define_region_from_file, mask_ne
from .ut import (
    compute_face_coords,
    get_company_name,
    get_data_2d,
    nodes_or_ements,
    update_attrs,
    match_longitude_format,
)


def lon_lat_to_cartesian(lon, lat, R=6371000):
    """
    Calculates Cartesian coordinates (x, y, z) from longitude and latitude coordinates
    on a sphere with radius R.

    Args:
        lon (float): The longitude coordinate in degrees.
        lat (float): The latitude coordinate in degrees.
        R (float, optional): The radius of the sphere. Defaults to 6371000 meters.

    Returns:
        tuple containing

        - x (float): The x-coordinate.
        - y (float): The y-coordinate.
        - z (float): The z-coordinate.

    References:
        - [Interpolation Between Grids with CKDTree](http://earthpy.org/interpolation_between_grids_with_ckdtree.html)
    """


    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x = R * np.cos(lat_r) * np.cos(lon_r)
    y = R * np.cos(lat_r) * np.sin(lon_r)
    z = R * np.sin(lat_r)
    return x, y, z


def create_indexes_and_distances(model_lon, model_lat, lons, lats, k=1, workers=2):
    """
    Creates a KDTree object and queries it for indexes of points in the FESOM mesh that are close to
    the points of the target grid. Also returns distances of the original points to target points.

    Parameters:
        model_lon (float): The longitude of the model points.
        model_lat (float): The latitude of the model points.
        lons (np.ndarray): 2D array with target grid longitudes.
        lats (np.ndarray): 2D array with target grid latitudes.
        k (int, optional): The number of nearest neighbors to return. Defaults to 1.
        workers (int, optional): Number of jobs to schedule for parallel processing. If -1 is given,
            all processors are used. Defaults to 2.

    Returns:        
        tuple containing

        - The distances to the nearest neighbors  (np.ndarray) 
        - The locations of the neighbors in the data (np.ndarray) 
    """

    xs, ys, zs = lon_lat_to_cartesian(model_lon, model_lat)
    xt, yt, zt = lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    tree = cKDTree(list(zip(xs, ys, zs)))
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k)

    return distances, inds


def ind_for_depth(depth, depths_from_file):
    """
    Finds the model depth index that is closest to the required depth.

    Parameters:
        depth (float): The desired depth.
        depths_from_file (List[float]): List of model depth levels.

    Returns:
        int: The index that corresponds to the model depth level closest to `depth`.

    """

    arr = [abs(abs(z) - abs(depth)) for z in depths_from_file]
    v, i = min((v, i) for (i, v) in enumerate(arr))
    dind = i
    return dind


def load_mesh(mesh_path):
    """
    Loads the mesh data from the specified path and returns the node coordinates and element connectivity.
    
    Args:
        mesh_path (str): The path to the directory containing the model output.

    Returns:
        tuple containing

        - x2 (np.ndarray): The x-coordinates of the mesh nodes.
        - y2 (np.ndarray): The y-coordinates of the mesh nodes.
        - elem (np.ndarray): The element connectivity array.

    """

    nodes = pd.read_csv(
        mesh_path + "/nod2d.out",
        delim_whitespace=True,
        skiprows=1,
        names=["node_number", "x", "y", "flag"],
    )

    x2 = nodes.x.values
    y2 = nodes.y.values

    file_content = pd.read_csv(
        mesh_path + "/elem2d.out",
        delim_whitespace=True,
        skiprows=1,
        names=["first_elem", "second_elem", "third_elem"],
    )

    x2 = np.where(x2 > 180, x2 - 360, x2)

    elem = file_content.values - 1

    return x2, y2, elem


def get_no_cyclic(x2, elem):
    """
    Computes the non-cyclic elements of the mesh.

    Args:
        x2 (np.ndarray): The x-coordinates of the mesh nodes.
        elem (np.ndarray): The element connectivity array.

    Returns:
        np.ndarray: An array containing the indices of the non-cyclic elements.

    """
    d = x2[elem].max(axis=1) - x2[elem].min(axis=1)
    no_cyclic_elem = np.argwhere(d < 100)
    return no_cyclic_elem.ravel()


def interpolate_kdtree2d(
    data_in,
    lons,
    distances,
    inds,
    radius_of_influence=100000,
    mask_zero=True,
):
    """

    Interpolates data using 2D KDTree interpolation.

    Args:
        data_in (np.ndarray): The input data array to be interpolated.
        lons (np.ndarray): The longitudes of the target grid points.
        distances (np.ndarray): The distances to the nearest neighbors.
        inds (np.ndarray): The locations of the neighbors in the data.
        radius_of_influence (float, optional): The radius of influence for interpolation.
            Data points with distances beyond this radius will be assigned NaN. Defaults to 100000.
        mask_zero (bool, optional): Flag indicating whether to mask zero values in the interpolated data
            by assigning them NaN. Defaults to True.


    Returns:
        np.ndarray: The interpolated data array with the same shape as the target grid.

    """
    interpolated = data_in[inds]
    interpolated[distances >= radius_of_influence] = np.nan
    if mask_zero:
        interpolated[interpolated == 0] = np.nan
    interpolated.shape = lons.shape

    return interpolated


def mask_triangulation(data_in, triang2, elem, no_cyclic_elem):
    """
    Applies mask on the triangulation object based on zero data values.

    Args:
        data_in (np.ndarray): Input data values.
        triang2 (tri.Triangulation): The triangulation object.
        elem (np.ndarray): The element connectivity (elem) array.
        no_cyclic_elem (np.ndarray): The non-cyclic elements of the mesh.

    Returns:
        tri.Triangulation: The masked triangulation object.
    """
    data_in_on_elements = data_in[elem[no_cyclic_elem]].mean(axis=1)
    data_in_on_elements[data_in_on_elements == 0] = -999
    mask = data_in_on_elements == -999
    triang2.set_mask(mask)
    return triang2


def interpolate_triangulation(
    data_in, 
    triang2, 
    trifinder,
    lon2, 
    lat2
):
    """
    Interpolates data using triangulation interpolation.

    Args:
        data_in (np.ndarray): The input data array to be interpolated.
        triang2 (mtri.Triangulation): The triangulation object.
        trifinder (Optional[mtri.TriFinder]): The triangulation finder. Defaults to None.
        lon2 (np.ndarray): The longitudes of the target grid points.
        lat2 (np.ndarray): The latitudes of the target grid points.

    Returns:
        np.ndarray: The interpolated data array.

    """
    interpolated = mtri.LinearTriInterpolator(triang2, data_in, trifinder=trifinder)(
        lon2, lat2
    )
    return interpolated


def interpolate_linear_scipy(data_in, x2, y2, lon2, lat2):
    """
    Interpolates data using linear interpolation with SciPy.

    Args:
        data_in (np.ndarray): The input data array to be interpolated.
        x2 (np.ndarray): The x-coordinates of the mesh nodes.
        y2 (np.ndarray): The y-coordinates of the mesh nodes.
        lon2 (np.ndarray): The longitudes of the target grid points.
        lat2 (np.ndarray): The latitudes of the target grid points.

    Returns:
        np.ndarray: The interpolated data array.

    """
    points = np.vstack((x2, y2)).T
    interpolated = LinearNDInterpolator(points, data_in)(lon2, lat2)
    return interpolated


def interpolate_cdo(target_grid,gridfile,original_file,output_file,variable_name,interpolation,weights_file_path, mask_zero=True):
    """
    Interpolate a climate variable in a NetCDF file using Climate Data Operators (CDO).

    Args:
        target_grid (str): Path to the target grid file (NetCDF format) for interpolation.
        gridfile (str): Path to the grid file (NetCDF format) associated with the original data.
        original_file (str): Path to the original NetCDF file containing the variable to be interpolated.
        output_file (str): Path to the output NetCDF file where the interpolated variable will be saved.
        variable_name (str): Name of the variable to be interpolated.
        interpolation (str): Interpolation method to be used (e.g., 'remapcon', 'remaplaf', 'remapnn', 'remapdis').
        weights_file_path (str): Path to the weights file generated by CDO for interpolation.
        mask_zero (bool, optional): Whether to mask zero values in the output. Default is True.

    Returns:
        np.ndarray: Interpolated variable data as a NumPy array.
    """
    command = [
        "cdo",
        f"-remap,{target_grid},{weights_file_path}",
        f"-setgrid,{gridfile}",
        f"{original_file}",
        f"{output_file}"
    ]
    if mask_zero:
        command.insert(1, "-setctomiss,0")

    # Execute the command
    subprocess.run(command)

    interpolated = xr.open_dataset(output_file)[variable_name].values
    os.remove(output_file)
    return interpolated

def generate_cdo_weights(target_grid,gridfile,original_file,output_file,interpolation, save = False):
    """
    Generate CDO interpolation weights for smmregrid and cdo interpolation.

    Args:
        target_grid (str): Path to the target grid file (NetCDF format).
        gridfile (str): Path to the grid file (NetCDF format) associated with the original data.
        original_file (str): Path to the original NetCDF file containing the data to be remapped.
        output_file (str): Path to the output NetCDF file where the weights will be saved.
        interpolation (str): Interpolation method to be used.
        save (bool, optional): Whether to save the weights file. Default is False.

    Returns:
        xr.Dataset: Generated interpolation weights as an xarray Dataset.
    """

    int_method = interpolation.split('_')[-1][5:]

    command = [
        "cdo",
        f"-gen{int_method},{target_grid}",
        f"-setgrid,{gridfile}",
        f"{original_file}",
        f"{output_file}"
    ]
    # Execute the command
    subprocess.run(command)

    weights = xr.open_dataset(output_file)
    if save == False:
        os.remove(output_file)

    return weights

def xesmf_weights_to_xarray(regridder):
    """
    Converts xESMF regridder weights to an xarray Dataset.

    This function takes a regridder object from xESMF, extracts the weights data,
    and converts it into an xarray Dataset with relevant dimensions and attributes.

    Args:
        regridder (xesmf.Regridder): A regridder object created using xESMF, which contains the weights to be converted.

    Returns:
        xr.Dataset: An xarray Dataset containing the weights data with dimensions 'n_s', 'col', and 'row'.
    """
    w = regridder.weights.data
    dim = 'n_s'
    ds = xr.Dataset(
        {
            'S': (dim, w.data),
            'col': (dim, w.coords[1, :] + 1),
            'row': (dim, w.coords[0, :] + 1),
        }
    )
    ds.attrs = {'n_in': regridder.n_in, 'n_out': regridder.n_out}
    return ds

def reconstruct_xesmf_weights(ds_w):
    """
    Reconstruct weights into a format that xESMF understands.

    This function takes a dataset with weights in a specific format and converts
    it into a format that can be used by xESMF for regridding.

    Args:
        ds_w (xarray.Dataset): The input dataset containing weights data.
            It should have 'S', 'col', 'row', and appropriate attributes 'n_out' and 'n_in'.

    Returns:
        xarray.DataArray: A DataArray containing reconstructed weights in COO format suitable for use with xESMF.
    """
    col = ds_w['col'].values - 1
    row = ds_w['row'].values - 1
    s = ds_w['S'].values
    n_out, n_in = ds_w.attrs['n_out'], ds_w.attrs['n_in']
    crds = np.stack([row, col])
    return xr.DataArray(
        sparse.COO(crds, s, (n_out, n_in)), dims=('out_dim', 'in_dim'), name='weights'
    )

def parse_depths(depths, depths_from_file):
    """
    Parses the selected depths from the available depth values and returns the corresponding depth indices and values.
    If depths = -1 returns all depths from depths_from_file

    Args:
        depths (Union[int, str]): The input depths specification.
            It can be an integer, a comma-separated list of integers,
            or a range specified with a colon (e.g., "10:100").
        depths_from_file (np.ndarray): The array of available depth values.

    Returns:
        tuple containing

        -The depth indices (List[int])
        -The depth values based on the input (List[int]) 

    """
    depth_type = "list"
    if len(depths.split(",")) > 1:
        depths = list(map(int, depths.split(",")))
    elif len(depths.split(":")) == 2:
        depth_min = int(depths.split(":")[0])
        depth_max = int(depths.split(":")[1])
        depth_type = "range"
    elif int(depths) == -1:
        depths = [-1]
    else:
        depths = [int(depths)]

    if depths[0] == -1:
        dinds = range(depths_from_file.shape[0])
        realdepths = depths_from_file
    else:
        dinds = []
        realdepths = []
        if depth_type == "list":
            for depth in depths:
                ddepth = ind_for_depth(depth, depths_from_file)
                dinds.append(ddepth)
                realdepths.append(depths_from_file[ddepth])
        elif depth_type == "range":
            ind_min = ind_for_depth(depth_min, depths_from_file)
            ind_max = ind_for_depth(depth_max, depths_from_file)
            for depth in np.array(depths_from_file)[ind_min:ind_max]:
                dinds.append(ind_for_depth(depth, depths_from_file))
                realdepths.append(depth)
    return dinds, realdepths


def parse_timesteps(timesteps, time_shape):
    """
    Parses the timesteps input and returns the corresponding timesteps as a list or range.

    Args:
        timesteps (Union[int, str]): The input timesteps specification.
            It can be an integer, a colon-separated range (e.g., "10:100"),
            a step range (e.g., "10:100:2"), or a comma-separated list of integers.
        time_shape (int): The total number of available timesteps.

    Returns:
        List[Union[int, range]]: The parsed timesteps as a list or range.

    """

    if len(timesteps.split(":")) == 2:
        y = range(int(timesteps.split(":")[0]), int(timesteps.split(":")[1]))
    elif len(timesteps.split(":")) == 3:
        if timesteps.split(":")[1] == "end":
            stop = time_shape
        else:
            stop = int(timesteps.split(":")[1])
        y = range(int(timesteps.split(":")[0]), stop, int(timesteps.split(":")[2]))
    elif len(timesteps.split(",")) > 1:
        y = list(map(int, timesteps.split(",")))
    elif int(timesteps) == -1:
        y = -1
    else:
        y = [int(timesteps)]
    timesteps = y
    print("timesteps {}".format(timesteps))
    return timesteps


def parse_timedelta(timedelta_arg):
    """
    Parses the timedelta argument and returns a numpy timedelta64 object.

    Args:
        timedelta_arg (str): The input timedelta argument in the format "<value><unit>".
            The value represents the magnitude of the timedelta, and the unit specifies
            the time unit (e.g., "s" for seconds, "m" for minutes, "h" for hours, "d" for days).

    Returns:
        np.timedelta64: The parsed numpy timedelta64 object.

    """
    timedelta_val = timedelta_arg[:-1]
    timedelta_unit = timedelta_arg[-1:]
    timedelta = np.timedelta64(timedelta_val, timedelta_unit)

    return timedelta


def save_data(
    data,
    args,
    timesteps,
    variable_name,
    interpolated3d,
    realdepths,
    x,
    y,
    lon,
    lat,
    out_path,
):
    """
    Saves the interpolated data to a NetCDF file.

    Args:
        data (xr.Dataset): The original dataset containing the input data, whose attributes will be taken.
        args (argparse.Namespace): Parsed command-line arguments.
        timesteps (Union[int, List[int]]): The indices or index of the selected timesteps to save.
        variable_name (str): The name of the interpolated variable.
        interpolated3d (np.ndarray): The interpolated 3D data array.
        realdepths (List[float]): The real depths corresponding to the interpolated depths.
        x (np.ndarray): The longitudes of the interpolated data.
        y (np.ndarray): The latitudes of the interpolated data.
        lon (np.ndarray): The longitudes of the original dataset.
        lat (np.ndarray): The latitudes of the original dataset.
        out_path (str): The path to save the output NetCDF file.

    Returns:
        None: This function does not return any value.

    """
    attributes = update_attrs(data.attrs, args)
    # if args.rotate:
    #     attributes2 = update_attrs(data2.attrs, args)
    data.attrs.update(attributes)
    # if args.rotate:
    #     data2.attrs.update(attributes2)
    if args.timedelta:
        timedelta = parse_timedelta(args.timedelta)
        shifted_time = data.time.data[timesteps] + timedelta
        out_time = np.atleast_1d(shifted_time)
    else:
        out_time = np.atleast_1d(data.time.data[timesteps])

    out1 = xr.Dataset(
        {variable_name: (["time", "depth", "lat", "lon"], interpolated3d)},
        coords={
            "time": out_time,
            "depth": realdepths,
            "lon": (["lon"], x),
            "lat": (["lat"], y),
            "longitude": (["lat", "lon"], lon),
            "latitude": (["lat", "lon"], lat),
        },
        attrs=data.attrs,
    )
    # if args.rotate:
    #     out2 = xr.Dataset(
    #         {variable_name2: (["time", "depth", "lat", "lon"], interpolated3d2)},
    #         coords={
    #             "time": out_time,
    #             "depth": realdepths,
    #             "lon": (["lon"], x),
    #             "lat": (["lat"], y),
    #             "longitude": (["lat", "lon"], lon),
    #             "latitude": (["lat", "lon"], lat),
    #         },
    #         attrs=data2.attrs,
    #     )

    # out1.to_netcdf(out_path, encoding={variable_name: {"zlib": True, "complevel": 9}})
    out1.to_netcdf(
        out_path,
        encoding={
            "time": {"dtype": np.dtype("double")},
            "depth": {"dtype": np.dtype("double")},
            "lat": {"dtype": np.dtype("double")},
            "lon": {"dtype": np.dtype("double")},
            "longitude": {"dtype": np.dtype("double")},
            "latitude": {"dtype": np.dtype("double")},
            variable_name: {"zlib": True, "complevel": 1, "dtype": np.dtype("single")},
        },
    )
    # if args.rotate:
    #     out2.to_netcdf(
    #         out_path2,
    #         encoding={
    #             "time": {"dtype": np.dtype("double")},
    #             "depth": {"dtype": np.dtype("double")},
    #             "lat": {"dtype": np.dtype("double")},
    #             "lon": {"dtype": np.dtype("double")},
    #             "longitude": {"dtype": np.dtype("double")},
    #             "latitude": {"dtype": np.dtype("double")},
    #             variable_name2: {
    #                 "zlib": True,
    #                 "complevel": 1,
    #                 "dtype": np.dtype("single"),
    #             },
    #         },
    #     )

    print(out1)


def fint(args=None):
    """
    Interpolates FESOM2 data to a regular grid.

    Args:
        args (argparse.Namespace, optional): Arguments from the command line. Defaults to None.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        prog="pfinterp", description="Interpolates FESOM2 data to regular grid."
    )
    parser.add_argument("data", help="Path to the FESOM2 output data file.")
    parser.add_argument("meshpath", help="Path to the FESOM2 mesh folder.")
    parser.add_argument(
        "--depths",
        "-d",
        default="0",
        type=str,
        help="Depths in meters. \
            Closest values from model levels will be taken.\
            Several options available: number - e.g. '100',\
                                       coma separated list - e.g. '0,10,100,200',\
                                       slice 0:100 \
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
        type=str,
        default="-180.0, 180.0, -80.0, 90.0",
        help="Several options are available:\
            - Map boundaries in -180 180 -90 90 format that will be used for interpolation.\
            - Use one of the predefined regions. Available: gs (Golf Stream), \
                trop (Atlantic Tropics), arctic, gulf (also Golf Stream, but based on Mercator projection.)))",
    )

    parser.add_argument(
        "--res",
        "-r",
        nargs=2,
        type=int,
        help="Number of points along each axis that will be used for interpolation (for lon and  lat).",
        metavar=("N_POINTS_LON", "N_POINTS_LAT"),
    )

    parser.add_argument(
        "--influence",
        "-i",
        default=80000,
        type=float,
        help="Radius of influence for interpolation, in meters. Used for nearest neighbor interpolation.",
    )

    parser.add_argument(
        "--map_projection",
        "-m",
        type=str,
        help="Map projection. Available: mer, np",
    )
    parser.add_argument(
        "--interp",
        choices=["nn", "mtri_linear", "linear_scipy",
                 "cdo_remapcon","cdo_remaplaf","cdo_remapnn", "cdo_remapdis",
                 "smm_remapcon","smm_remaplaf","smm_remapnn", "smm_remapdis", "xesmf_nearest_s2d"],  # "idist", "linear", "cubic"],
        default="nn",
        help="Interpolation method. Options are \
            nn - nearest neighbor (KDTree implementation, fast), \
            mtri_linear - linear, based on triangulation information (slow, but more precise)",
    )

    parser.add_argument(
        "--mask",
        type=str,
        help="File with mask for interpolation. Mask should have the same coordinates as interpolated data. \
            Usuall usecase is to use mtri_linear slow interpolation to create the mask, and then use this mask for faster (nn) interpolation.",
    )

    parser.add_argument(
        "--ofile",
        "-o",
        type=str,
        help="Path to the output file. Default is out.nc.",
    )
    parser.add_argument(
        "--odir",
        default="./",
        type=str,
        help="Path to the output directory. Default is ./",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Path to the target file, that contains coordinates of the target grid (as lon lat variables)",
    )

    parser.add_argument(
        "--no_shape_mask",
        action="store_true",
        help="Do not apply shapely mask for coastlines. Useful for paleo applications.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate vector variables to geographic coordinates. Use standard FESOM2 angles (this should be fine in 99.99 percent of cases:)).",
    )

    parser.add_argument(
        "--no_mask_zero",
        action="store_false",
        help="FESOM2 use 0 as mask value, which is terrible for some variables. \
            Solution is to create a mask using temperature or salinity, and then use this mask for interpolation, applying this option.",
    )

    parser.add_argument(
        "--timedelta",
        type=str,
        help="Add timedelta to the time axis. The format is number followed by unit. E.g. '1D' or '10h'. \
              Valid units are 'D' (days), 'h' (hours), 'm' (minutes), 's' (seconds). \
              To substract timedelta, put argument in quotes, and prepend ' -', so SPACE and then -, e.g. ' -10D'.",
    )
    parser.add_argument(
        "--oneout",
        action="store_true",
        help="Add timedelta to the time axis. The format is number followed by unit. E.g. '1D' or '10h'. \
              Valid units are 'D' (days), 'h' (hours), 'm' (minutes), 's' (seconds). \
              To substract timedelta, put argument in quotes, and prepend ' -', so SPACE and then -, e.g. ' -10D'.",
    )
    parser.add_argument(
        "--weightspath",
        type=str,
        help="File with CDO weiths for smm interpolation.")
    parser.add_argument(
        "--save_weights",
        action = "store_true",
        help   = "Save CDO weights ac netcdf file at the output directory.")
    

    args = parser.parse_args()

    # we will extract some arguments and will not pass just args to function,
    # because we should keep functions extractable from the code.
    data = xr.open_dataset(args.data)
    variable_name = list(data.data_vars)[0]
    data_file = os.path.basename(args.data)
    if args.rotate:
        data_file_ori = data_file
        variable_name_orig = variable_name
        company_name = get_company_name(variable_name)
        print(company_name)
        variable_name = company_name[0]
        variable_name2 = company_name[1]
        data_file = data_file_ori.replace(variable_name_orig, variable_name)
        data_file2 = data_file_ori.replace(variable_name_orig, variable_name2)
        data = xr.open_dataset(args.data.replace(data_file_ori, data_file))
        data2 = xr.open_dataset(args.data.replace(data_file_ori, data_file2))
    radius_of_influence = args.influence
    projection = args.map_projection

    # not the most elegant way, but let's assume that we have only one variable

    dim_names = list(data.coords)
    interpolation = args.interp
    mask_file = args.mask
    out_file = args.ofile

    # open mask file if needed
    if mask_file is not None:
        mask_data = xr.open_dataset(mask_file)
        # again assume we have only one variable inside
        mask_variable_name = list(mask_data.data_vars)[0]
        mask_data = mask_data[mask_variable_name]

    # if data have depth, parse depths
    if ("nz1" in data.dims) or ("nz" in data.dims):
        if "nz1" in data.dims:
            depth_coord = "nz1"
        else:
            depth_coord = "nz"
        depths_from_file = data[depth_coord].values
        dinds, realdepths = parse_depths(args.depths, depths_from_file)
        if (data[variable_name].dims[1] == "nz") or (
            data[variable_name].dims[1] == "nz1"
        ):
            dimension_order = "normal"
        else:
            dimension_order = "transpose"
    else:
        dinds = [0]
        realdepths = [0]
        dimension_order = "normal"

    # prepear time steps
    time_shape = data.time.shape[0]
    timesteps = parse_timesteps(args.timesteps, time_shape)
    if timesteps == -1:
        timesteps = range(time_shape)
    # set timestep to 0 if data have only one time step
    if time_shape == 1:
        timesteps = [0]

    # prepear output file name and path
    if out_file is None:
        output_file = os.path.basename(data_file)
        if args.rotate:
            output_file2 = os.path.basename(data_file2)

        if args.target is None:
            region = args.box.replace(", ", "_")
        else:
            region = os.path.basename(args.target)
        out_file = output_file.replace(
            ".nc",
            f"_interpolated_{region}_{realdepths[0]}_{realdepths[-1]}_{timesteps[0]}_{timesteps[-1]}.nc",
        )
        if args.rotate:
            out_file2 = output_file2.replace(
                ".nc",
                f"_interpolated_{region}_{realdepths[0]}_{realdepths[-1]}_{timesteps[0]}_{timesteps[-1]}.nc",
            )
        out_path = os.path.join(args.odir, out_file)
        if args.rotate:
            out_path2 = os.path.join(args.odir, out_file2)
    else:
        out_path = os.path.join(args.odir, out_file)
        if args.rotate:
            out_file2 = out_file.replace(".nc", "_2.nc")
            out_path2 = os.path.join(args.odir, out_file2)

    x2, y2, elem = load_mesh(args.meshpath)

    placement = nodes_or_ements(data, variable_name, len(x2), len(elem))
    if placement == "elements":
        if args.interp == "mtri_linear":
            raise ValueError("mtri_linear interpolation is not supported for elements")
        face_x, face_y = compute_face_coords(x2, y2, elem)
        x2, y2 = face_x, face_y

    # define region of interpolation
    if args.target is None:
        x, y, lon, lat = define_region(args.box, args.res, projection)
    else:
        x, y, lon, lat = define_region_from_file(args.target)

    x2, y2 = match_longitude_format(x2, y2, lon, lat)
    # if we want to use shapelly mask, load it
    if args.no_shape_mask is False:
        m2 = mask_ne(lon, lat)

    # additional variables, that we need for different interplations
    if interpolation == "mtri_linear":
        no_cyclic_elem = get_no_cyclic(x2, elem)
        triang2 = mtri.Triangulation(x2, y2, elem[no_cyclic_elem])
        trifinder = triang2.get_trifinder()
    elif interpolation == "nn":
        distances, inds = create_indexes_and_distances(x2, y2, lon, lat, k=1, workers=4)
    elif interpolation in ['xesmf_nearest_s2d']:
        ds_in = xr.open_dataset(args.data)
        ds_in = ds_in.assign_coords(lat=('nod2',y2), lon=('nod2',x2))
        ds_in.lat.attrs = {'units': 'degrees', 'standard_name': 'latitude'}
        ds_in.lon.attrs = {'units': 'degrees', 'standard_name': 'longitude'}
        ds_out = xr.Dataset(
        {
            'x': xr.DataArray(x, dims=['x']),
            'y': xr.DataArray(y, dims=['y']),
            'lat': xr.DataArray(lat, dims=['y', 'x']),
            'lon': xr.DataArray(lon, dims=['y', 'x']),
        })
        if args.weightspath is not None:
            xesmf_weights_path = args.weightspath
            ds_w = xr.open_dataset(xesmf_weights_path)
            wegiths_xesmf = reconstruct_xesmf_weights(ds_w)
            regridder = xe.Regridder(ds_in,ds_out, method='nearest_s2d', weights=wegiths_xesmf,locstream_in=True)
        else:
            regridder = xe.Regridder(ds_in, ds_out, method='nearest_s2d', locstream_in=True)
        if args.save_weights is True:
            ds_w = xesmf_weights_to_xarray(regridder)
            xesmf_weights_path = out_path.replace(".nc", "xesmf_weights.nc")
            ds_w.to_netcdf(xesmf_weights_path)
    elif interpolation in ["cdo_remapcon", "cdo_remaplaf", "cdo_remapnn", "cdo_remapdis", "smm_remapcon", "smm_remaplaf", "smm_remapnn", "smm_remapdis"]:
        gridtype = 'latlon'
        gridsize = x.size*y.size
        xsize = x.size
        ysize = y.size
        xname = 'longitude'
        xlongname = 'longitude'
        xunits = 'degrees_east'
        yname = 'latitude'
        ylongname = 'latitude'
        yunits = 'degrees_north'
        xfirst = float(lon[0,0])
        xinc = float(lon[0,1]-lon[0,0])
        yfirst = float(lat[0,0])
        yinc = float(lat[1,0]-lat[0,0])
        grid_mapping = []
        grid_mapping_name = []
        straight_vertical_longitude_from_pole = []
        latitude_of_projection_origin = []
        standard_parallel = []
        if projection == "np":
            gridtype = 'projection'
            xlongname = 'x coordinate of projection'
            xunits = 'meters'
            ylongname = 'y coordinate of projection'
            yunits = 'meters'
            xfirst = float(x[0])
            xinc = float(x[1]-x[0])
            yfirst = float(y[0])
            yinc = float(y[1]-y[0])
            grid_mapping = 'crs'
            grid_mapping_name = 'polar_stereographic'
            straight_vertical_longitude_from_pole = 0.0
            latitude_of_projection_origin = 90.0
            standard_parallel = 71.0
        
        
        formatted_content = f"""\
        gridtype = {gridtype}
        gridsize = {gridsize}
        xsize = {xsize}
        ysize = {ysize}
        xname = {xname}
        xlongname = "{xlongname}"
        xunits = "{xunits}"
        yname = {yname}
        ylongname = "{ylongname}"
        yunits = "{yunits}"
        xfirst = {xfirst}
        xinc = {xinc}
        yfirst = {yfirst}
        yinc = {yinc}
        grid_mapping = {grid_mapping}
        grid_mapping_name = {grid_mapping_name}
        straight_vertical_longitude_from_pole = {straight_vertical_longitude_from_pole}
        latitude_of_projection_origin = {latitude_of_projection_origin}
        standard_parallel = {standard_parallel}"""
        
        target_grid_path = out_path.replace(".nc", "target_grid.txt")
        with open(target_grid_path, 'w') as file:
            file.write(formatted_content)
        
        endswith = "_nodes_IFS.nc"
        if placement == "elements":
            endswith = "_elements_IFS.nc"
        for root, dirs, files in os.walk(args.meshpath):
            for file in files:
                if file.endswith(endswith):
                    gridfile = os.path.join(root, file)


    # we will fill this array with interpolated values
    if not args.oneout:
        interpolated3d = np.zeros((len(timesteps), len(realdepths), len(y), len(x)))
        if args.rotate:
            interpolated3d2 = np.zeros(
                (len(timesteps), len(realdepths), len(y), len(x))
            )
    else:
        interpolated3d = np.zeros((1, len(realdepths), len(y), len(x)))
        if args.rotate:
            interpolated3d2 = np.zeros((1, len(realdepths), len(y), len(x)))

    # main loop
    for t_index, ttime in enumerate(timesteps):
        for d_index, (dind, realdepth) in enumerate(zip(dinds, realdepths)):
            print(f"time: {ttime}, depth:{realdepth}")

            if args.rotate:
                data_in, data_in2 = get_data_2d(
                    [data, data2],
                    [variable_name, variable_name2],
                    ttime,
                    dind,
                    dimension_order,
                    args.rotate,
                    x2,
                    y2,
                )
            else:
                data_in = get_data_2d(
                    [data],
                    [variable_name],
                    ttime,
                    dind,
                    dimension_order,
                    args.rotate,
                    x2,
                    y2,
                )

            if interpolation == "mtri_linear":
                # we don't use shapely mask with this method
                args.no_shape_mask = True
                if mask_file is None:
                    triang2 = mask_triangulation(data_in, triang2, elem, no_cyclic_elem)
                    if args.rotate:
                        triang2_2 = mask_triangulation(
                            data_in2, triang2, elem, no_cyclic_elem
                        )
                interpolated = interpolate_triangulation(
                    data_in, triang2, trifinder,lon, lat
                )

                if args.rotate:
                    interpolated2 = interpolate_triangulation(
                        data_in2,
                        triang2_2,
                        trifinder,
                        lon,
                        lat
                    )
            elif interpolation == "nn":
                interpolated = interpolate_kdtree2d(
                    data_in,
                    lon,
                    distances,
                    inds,
                    radius_of_influence=radius_of_influence,
                    mask_zero=args.no_mask_zero,
                )
                if args.rotate:
                    interpolated2 = interpolate_kdtree2d(
                        data_in2,
                        lon,
                        distances,
                        inds,
                        radius_of_influence=radius_of_influence,
                        mask_zero=args.no_mask_zero,
                    )
            elif interpolation == "linear_scipy":
                interpolated = interpolate_linear_scipy(data_in, x2, y2, lon, lat)
                if args.rotate:
                    interpolated2 = interpolate_linear_scipy(data_in2, x2, y2, lon, lat)
            
            elif interpolation in ["cdo_remapcon", "cdo_remaplaf", "cdo_remapnn", "cdo_remapdis"]:
                input_data = xr.Dataset({variable_name: (["nod2"], data_in)})
                if args.rotate:
                    input_data = xr.Dataset({variable_name: (["nod2"], data_in2)})
                output_file_path = out_path.replace(".nc", "output_cdo_file.nc")
                input_file_path = args.data.replace(".nc","cdo_interpolation.nc")
                input_data.to_netcdf(input_file_path,encoding={
                            variable_name: {"dtype": np.dtype("double")},
                        },
                    )
                if args.weightspath is not None:
                        weights_file_path = args.weightspath
                else:
                    if t_index == 0 and d_index == 0:
                        weights_file_path = out_path.replace(".nc", "weighs_cdo.nc")
                        weights = generate_cdo_weights(target_grid_path,
                                gridfile,
                                input_file_path,
                                weights_file_path,
                                interpolation,
                                save = True)
                interpolated = interpolate_cdo(target_grid_path,
                                               gridfile,
                                               input_file_path,
                                               output_file_path,
                                               variable_name,
                                                interpolation,
                                               weights_file_path,
                                                mask_zero=args.no_mask_zero
                                                )
                os.remove(input_file_path)

            elif interpolation in ["smm_remapcon", "smm_remaplaf", "smm_remapnn", "smm_remapdis"]:
                input_data = xr.Dataset({variable_name: (["nod2"], data_in)})
                if args.rotate:
                    input_data = xr.Dataset({variable_name: (["nod2"], data_in2)})
                if t_index == 0 and d_index == 0:
                    input_file_path = args.data.replace(".nc","cdo_interpolation.nc")
                    input_data.to_netcdf(input_file_path,encoding={
                                variable_name: {"dtype": np.dtype("double")},
                            },
                        )
                    output_file_path = out_path.replace(".nc", "weighs_cdo.nc")
                    if args.weightspath is not None:
                        weights_file = args.weightspath
                        weights = xr.open_dataset(weights_file)
                    else:
                        weights = generate_cdo_weights(target_grid_path,
                                                    gridfile,
                                                    input_file_path,
                                                    output_file_path,
                                                    interpolation,
                                                    save =args.save_weights)
                    os.remove(input_file_path)
                    interpolator = Regridder(weights=weights)
                interpolated = interpolator.regrid(input_data)
                interpolated = interpolated[variable_name].values
                mask_zero=args.no_mask_zero
                if mask_zero:
                    interpolated[interpolated == 0] = np.nan
            
            elif interpolation in ['xesmf_nearest_s2d']:
                ds_in = xr.Dataset({variable_name: (["nod2"], data_in)})
                ds_in = ds_in.assign_coords(lat=('nod2',y2), lon=('nod2',x2))
                ds_in.lat.attrs = {'units': 'degrees', 'standard_name': 'latitude'}
                ds_in.lon.attrs = {'units': 'degrees', 'standard_name': 'longitude'}
                interpolated = regridder(ds_in)[variable_name].values

            # masking of the data
            if mask_file is not None:
                mask_level = mask_data[0, dind, :, :].values
                mask = np.ma.masked_invalid(mask_level).mask
                interpolated[mask] = np.nan
                if args.rotate:
                    interpolated2[mask] = np.nan
            elif args.no_shape_mask is False:
                interpolated[m2] = np.nan
                if args.rotate:
                    interpolated2[m2] = np.nan

            if args.oneout:
                interpolated3d[0, d_index, :, :] = interpolated
                if args.rotate:
                    interpolated3d2[0, d_index, :, :] = interpolated2
            else:
                interpolated3d[t_index, d_index, :, :] = interpolated
                if args.rotate:
                    interpolated3d2[t_index, d_index, :, :] = interpolated2

        if args.oneout:
            out_path_one = out_path.replace(".nc", f"_{str(t_index).zfill(10)}.nc")
            save_data(
                data,
                args,
                [ttime],
                variable_name,
                interpolated3d,
                realdepths,
                x,
                y,
                lon,
                lat,
                out_path_one,
            )
            if args.rotate:
                out_path_one2 = out_path2.replace(
                    ".nc", f"_{str(t_index).zfill(10)}.nc"
                )
                save_data(
                    data2,
                    args,
                    [ttime],
                    variable_name2,
                    interpolated3d2,
                    realdepths,
                    x,
                    y,
                    lon,
                    lat,
                    out_path_one2,
                )
    if interpolation in ["cdo_remapcon","cdo_remaplaf","cdo_remapnn","cdo_remapdis"]:    
        os.remove(target_grid_path)
        if args.save_weights is False and args.weightspath is None and weights_file_path is not None:
            os.remove(weights_file_path)
    elif interpolation in ["smm_remapcon","smm_remapnn","smm_remaplaf","smm_remapdis"]:
        os.remove(target_grid_path)

    # save data (always 4D array)
    if not args.oneout:
        save_data(
            data,
            args,
            timesteps,
            variable_name,
            interpolated3d,
            realdepths,
            x,
            y,
            lon,
            lat,
            out_path,
        )
        if args.rotate:
            save_data(
                data2,
                args,
                timesteps,
                variable_name2,
                interpolated3d2,
                realdepths,
                x,
                y,
                lon,
                lat,
                out_path2,
            )


if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    fint()
