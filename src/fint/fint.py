from platform import node
import xarray as xr
import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    NearestNDInterpolator,
)
import matplotlib.tri as mtri
import pandas as pd
from scipy.spatial import cKDTree
import argparse
from .regions import define_region, define_region_from_file, mask_ne
import os
from .ut import (
    update_attrs,
    nodes_or_ements,
    compute_face_coords,
    get_company_name,
    get_data_2d,
)


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
    distances, inds = tree.query(list(zip(xt, yt, zt)), k=k)

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
    """Compute non cyclic elements of the mesh."""
    d = x2[elem].max(axis=1) - x2[elem].min(axis=1)
    no_cyclic_elem = np.argwhere(d < 100)
    return no_cyclic_elem.ravel()


def interpolate_kdtree2d(
    data_in,
    x2,
    y2,
    elem,
    lons,
    lats,
    distances,
    inds,
    radius_of_influence=100000,
    mask_zero=True,
):

    interpolated = data_in[inds]
    interpolated[distances >= radius_of_influence] = np.nan
    if mask_zero:
        interpolated[interpolated == 0] = np.nan
    interpolated.shape = lons.shape

    return interpolated


def mask_triangulation(data_in, triang2, elem, no_cyclic_elem):
    data_in_on_elements = data_in[elem[no_cyclic_elem]].mean(axis=1)
    data_in_on_elements[data_in_on_elements == 0] = -999
    mmask = data_in_on_elements == -999
    data_in[data_in == 0] = np.nan
    triang2.set_mask(mmask)
    return triang2


def interpolate_triangulation(
    data_in, triang2, trifinder, x2, y2, lon2, lat2, elem, no_cyclic_elem
):
    interpolated = mtri.LinearTriInterpolator(triang2, data_in, trifinder=trifinder)(
        lon2, lat2
    )
    return interpolated


def interpolate_linear_scipy(data_in, x2, y2, lon2, lat2):
    points = np.vstack((x2, y2)).T
    interpolated = LinearNDInterpolator(points, data_in)(lon2, lat2)
    return interpolated


def parse_depths(depths, depths_from_file):
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
    timedelta_val = timedelta_arg[:-1]
    timedelta_unit = timedelta_arg[-1:]
    timedelta = np.timedelta64(timedelta_val, timedelta_unit)

    return timedelta

def save_data(data, args, timesteps, variable_name, interpolated3d, realdepths, x, y, lon, lat, out_path):
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
                trop (Atlantic Tropics), arctic, gulf (also Golf Stream, but based on Mercator projection.)\))",
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
        choices=["nn", "mtri_linear", "linear_scipy"],  # "idist", "linear", "cubic"],
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
              To substract timedelta, put argument in quotes, and prepend ' -', so SPACE and then -, e.g. ' -10D'."
    )
    parser.add_argument(
        "--oneout",
        action="store_true",
        help="Add timedelta to the time axis. The format is number followed by unit. E.g. '1D' or '10h'. \
              Valid units are 'D' (days), 'h' (hours), 'm' (minutes), 's' (seconds). \
              To substract timedelta, put argument in quotes, and prepend ' -', so SPACE and then -, e.g. ' -10D'."
    )

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
        depth_coord = dim_names[0]
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

    # if we want to use shapelly mask, load it
    if args.no_shape_mask is False:
        m2 = mask_ne(lon, lat)

    # additional variables, that we need for sifferent interplations
    if interpolation == "mtri_linear":
        no_cyclic_elem = get_no_cyclic(x2, elem)
        triang2 = mtri.Triangulation(x2, y2, elem[no_cyclic_elem])
        trifinder = triang2.get_trifinder()
    elif interpolation == "nn":
        distances, inds = create_indexes_and_distances(x2, y2, lon, lat, k=1, workers=4)

    # we will fill this array with interpolated values
    if not args.oneout:
        interpolated3d = np.zeros((len(timesteps), len(realdepths), len(y), len(x)))
        if args.rotate:
            interpolated3d2 = np.zeros((len(timesteps), len(realdepths), len(y), len(x)))
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
                    data_in, triang2, trifinder, x2, y2, lon, lat, elem, no_cyclic_elem
                )
                if args.rotate:
                    interpolated2 = interpolate_triangulation(
                        data_in2,
                        triang2_2,
                        trifinder,
                        x2,
                        y2,
                        lon,
                        lat,
                        elem,
                        no_cyclic_elem,
                    )
            elif interpolation == "nn":
                interpolated = interpolate_kdtree2d(
                    data_in,
                    x2,
                    y2,
                    elem,
                    lon,
                    lat,
                    distances,
                    inds,
                    radius_of_influence=radius_of_influence,
                    mask_zero=args.no_mask_zero,
                )
                if args.rotate:
                    interpolated2 = interpolate_kdtree2d(
                        data_in2,
                        x2,
                        y2,
                        elem,
                        lon,
                        lat,
                        distances,
                        inds,
                        radius_of_influence=radius_of_influence,
                        mask_zero=args.no_mask_zero,
                    )
            elif interpolation == "linear_scipy":
                interpolated = interpolate_linear_scipy(data_in, x2, y2, lon, lat)
                if args.rotate:
                    interpolated2 = interpolate_linear_scipy(data_in2, x2, y2, lon, lat)

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
            out_path_one = out_path.replace('.nc', f'_{str(t_index).zfill(10)}.nc')
            save_data(data, args, [ttime], variable_name, interpolated3d, realdepths, x, y, lon, lat, out_path_one)
            if args.rotate:
                out_path_one2 = out_path2.replace('.nc', f'_{str(t_index).zfill(10)}.nc')
                save_data(data2, args, [ttime], variable_name2, interpolated3d2, realdepths, x, y, lon, lat, out_path_one2)        

    # save data (always 4D array)
    if not args.oneout:
        save_data(data, args, timesteps, variable_name, interpolated3d, realdepths, x, y, lon, lat, out_path)
        if args.rotate:
            save_data(data2, args, timesteps, variable_name2, interpolated3d2, realdepths, x, y, lon, lat, out_path2)




if __name__ == "__main__":
    # args = parser.parse_args()
    # args.func(args)
    fint()
