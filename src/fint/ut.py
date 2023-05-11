import os

import numpy as np


def update_attrs(ds, args):
    """
    Updates attributes of the dataset with values from the arguments.

    Args:
        ds (dict): The dataset attributes dictionary to update.
        args (argparse.Namespace): The arguments containing attribute values.

    Returns:
        dict: The updated dataset attributes dictionary.

    """
    if args.target is not None:
        ds["target"] = args.target

    ds["box"] = args.box
    ds["influence"] = args.influence
    ds["interp"] = args.interp
    ds["data"] = os.path.abspath(args.data)
    ds["meshpath"] = os.path.abspath(args.meshpath)
    return ds


def nodes_or_ements(data, variable_name, node_num, elem_num):
    """
    Determines if a variable in the data corresponds to nodes or elements.

    Args:
        data (dict): The data dictionary.
        variable_name (str): The name of the variable to check.
        node_num (int): The expected number of nodes.
        elem_num (int): The expected number of elements.

    Returns:
        str: Either "nodes" or "elements" based on the shape of the variable.

    """


    if data[variable_name].shape[-1] == node_num:
        return "nodes"
    elif data[variable_name].shape[-1] == elem_num:
        return "elements"


def compute_face_coords(x2, y2, elem):
    """
    Compute the coordinates of the elements (triangles).

    Args:
        x2 (numpy array): X coordinates.
        y2 (numpy array): Y coordinates.
        elem (numpy array): Element indices.

    Returns:
        - numpy array: face_x, representing the computed x coordinates of the elements
        - numpy array: face_y, representing the computed y coordinates of the elements
    """
    first_mean = x2[elem].mean(axis=1)
    j = np.where(np.abs(x2[elem][:, 0] - first_mean) > 100)[0]
    cyclic_elems = x2[elem][j].copy()
    new_means = np.where(cyclic_elems > 0, cyclic_elems, cyclic_elems + 360).mean(
        axis=1
    )
    new_means[new_means > 180] = new_means[new_means > 180] - 360
    first_mean[j] = new_means
    face_x = first_mean
    face_y = y2[elem].mean(axis=1)
    return face_x, face_y

def get_company_name(variable_name):
    """
    Get the corresponding vector variables for a given variable name.

    Args:
        variable_name (str): The variable name.

    Returns:
        list: A list of corresponding vector variables.

    Raises:
        KeyError: If the variable name is not found in the dictionary.
    """
    vector_vars = {
        "u": ["u", "v"],
        "v": ["u", "v"],
        "uice": ["uice", "vice"],
        "vice": ["uice", "vice"],
        "unod": ["unod", "vnod"],
        "vnod": ["unod", "vnod"],
        "tau_x": ["tau_x", "tau_y"],
        "tau_y": ["tau_x", "tau_y"],
        "atmice_x": ["atmice_x", "atmice_y"],
        "atmice_y": ["atmice_x", "atmice_x"],
        "atmoce_x": ["atmoce_x", "atmoce_y"],
        "atmoce_y": ["atmoce_x", "atmoce_y"],
        "iceoce_x": ["iceoce_x", "iceoce_y"],
        "iceoce_y": ["iceoce_x", "iceoce_y"],
        "tx_sur": ["tx_sur", "ty_sur"],
        "ty_sur": ["tx_sur", "ty_sur"],
    }

    return vector_vars[variable_name]


def scalar_g2r(al, be, ga, lon, lat):
    """
    Converts geographical coordinates to rotated coordinates.

    Parameters:
        al (float): Alpha Euler angle.
        be (float): Beta Euler angle.
        ga (float): Gamma Euler angle.
        lon (array): 1D array of longitudes in geographical coordinates.
        lat (array): 1D array of latitudes in geographical coordinates.

    Returns:
        - array: 1D array of longitudes in rotated coordinates.
        - array: 1D array of latitudes in rotated coordinates.
    """

    rad = np.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad

    rotate_matrix = np.zeros(shape=(3, 3))

    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    # rotate_matrix = np.linalg.pinv(rotate_matrix)

    lat = lat * rad
    lon = lon * rad

    # geographical Cartesian coordinates:
    xr = np.cos(lat) * np.cos(lon)
    yr = np.cos(lat) * np.sin(lon)
    zr = np.sin(lat)

    # rotated Cartesian coordinates:
    xg = rotate_matrix[0, 0] * xr + rotate_matrix[0, 1] * yr + rotate_matrix[0, 2] * zr
    yg = rotate_matrix[1, 0] * xr + rotate_matrix[1, 1] * yr + rotate_matrix[1, 2] * zr
    zg = rotate_matrix[2, 0] * xr + rotate_matrix[2, 1] * yr + rotate_matrix[2, 2] * zr

    # rotated coordinates:
    rlat = np.arcsin(zg)
    rlon = np.arctan2(yg, xg)

    a = np.where((np.abs(xg) + np.abs(yg)) == 0)
    if a:
        lon[a] = 0

    rlat = rlat / rad
    rlon = rlon / rad

    return (rlon, rlat)


def scalar_r2g(al, be, ga, rlon, rlat):
    """
    Converts rotated coordinates to geographical coordinates.

    Parameters:
        al (float): Alpha Euler angle.
        be (float): Beta Euler angle.
        ga (float): Gamma Euler angle.
        rlon (array): 1D array of longitudes in rotated coordinates.
        rlat (array): 1D array of latitudes in rotated coordinates.

    Returns:
        - array: 1D array of longitudes in geographical coordinates.
        - array: 1D array of latitudes in geographical coordinates.
    """

    rad = np.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad
    rotate_matrix = np.zeros(shape=(3, 3))
    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    rotate_matrix = np.linalg.pinv(rotate_matrix)

    rlat = rlat * rad
    rlon = rlon * rad

    # Rotated Cartesian coordinates:
    xr = np.cos(rlat) * np.cos(rlon)
    yr = np.cos(rlat) * np.sin(rlon)
    zr = np.sin(rlat)

    # Geographical Cartesian coordinates:
    xg = rotate_matrix[0, 0] * xr + rotate_matrix[0, 1] * yr + rotate_matrix[0, 2] * zr
    yg = rotate_matrix[1, 0] * xr + rotate_matrix[1, 1] * yr + rotate_matrix[1, 2] * zr
    zg = rotate_matrix[2, 0] * xr + rotate_matrix[2, 1] * yr + rotate_matrix[2, 2] * zr

    # Geographical coordinates:
    lat = np.arcsin(zg)
    lon = np.arctan2(yg, xg)

    a = np.where((np.abs(xg) + np.abs(yg)) == 0)
    if a:
        lon[a] = 0

    lat = lat / rad
    lon = lon / rad

    return (lon, lat)


def vec_rotate_r2g(al, be, ga, lon, lat, urot, vrot, flag):
    """
    Rotates vectors from rotated coordinates to geographical coordinates.

    Parameters:
        al (float): Alpha Euler angle.
        be (float): Beta Euler angle.
        ga (float): Gamma Euler angle.
        lon (array): 1D array of longitudes in rotated or geographical coordinates (see flag parameter).
        lat (array): 1D array of latitudes in rotated or geographical coordinates (see flag parameter).
        urot (array): 1D array of u component of the vector in rotated coordinates.
        vrot (array): 1D array of v component of the vector in rotated coordinates.
        flag (int): Flag indicating the coordinate system of lon and lat:
            - flag = 1: lon and lat are in geographical coordinates.
            - flag = 0: lon and lat are in rotated coordinates.

    Returns:
        - array: 1D array of u component of the vector in geographical coordinates.
        - array: 1D array of v component of the vector in geographical coordinates.
    """

    #   first get another coordinate
    if flag == 1:
        (rlon, rlat) = scalar_g2r(al, be, ga, lon, lat)
    else:
        rlon = lon
        rlat = lat
        (lon, lat) = scalar_r2g(al, be, ga, rlon, rlat)

    #   then proceed...
    rad = np.pi / 180
    al = al * rad
    be = be * rad
    ga = ga * rad

    rotate_matrix = np.zeros(shape=(3, 3))
    rotate_matrix[0, 0] = np.cos(ga) * np.cos(al) - np.sin(ga) * np.cos(be) * np.sin(al)
    rotate_matrix[0, 1] = np.cos(ga) * np.sin(al) + np.sin(ga) * np.cos(be) * np.cos(al)
    rotate_matrix[0, 2] = np.sin(ga) * np.sin(be)
    rotate_matrix[1, 0] = -np.sin(ga) * np.cos(al) - np.cos(ga) * np.cos(be) * np.sin(
        al
    )
    rotate_matrix[1, 1] = -np.sin(ga) * np.sin(al) + np.cos(ga) * np.cos(be) * np.cos(
        al
    )
    rotate_matrix[1, 2] = np.cos(ga) * np.sin(be)
    rotate_matrix[2, 0] = np.sin(be) * np.sin(al)
    rotate_matrix[2, 1] = -np.sin(be) * np.cos(al)
    rotate_matrix[2, 2] = np.cos(be)

    rotate_matrix = np.linalg.pinv(rotate_matrix)

    rlat = rlat * rad
    rlon = rlon * rad
    lat = lat * rad
    lon = lon * rad

    #   vector in rotated Cartesian
    txg = -vrot * np.sin(rlat) * np.cos(rlon) - urot * np.sin(rlon)
    tyg = -vrot * np.sin(rlat) * np.sin(rlon) + urot * np.cos(rlon)
    tzg = vrot * np.cos(rlat)

    #   vector in geo Cartesian
    txr = (
        rotate_matrix[0, 0] * txg
        + rotate_matrix[0, 1] * tyg
        + rotate_matrix[0, 2] * tzg
    )
    tyr = (
        rotate_matrix[1, 0] * txg
        + rotate_matrix[1, 1] * tyg
        + rotate_matrix[1, 2] * tzg
    )
    tzr = (
        rotate_matrix[2, 0] * txg
        + rotate_matrix[2, 1] * tyg
        + rotate_matrix[2, 2] * tzg
    )

    #   vector in geo coordinate
    v = (
        -np.sin(lat) * np.cos(lon) * txr
        - np.sin(lat) * np.sin(lon) * tyr
        + np.cos(lat) * tzr
    )
    u = -np.sin(lon) * txr + np.cos(lon) * tyr

    u = np.array(u)
    v = np.array(v)

    return (u, v)


def get_data_2d(datas, variable_names, ttime, dind, dimension_order, rotate, x2, y2):
    """
    Get 2D data from a multidimensional dataset and optionally rotate the data.

    Parameters:
        datas (list): List of dataset objects.
        variable_names (list): List of variable names.
        ttime (int): Time index.
        dind (int): Dimension index.
        dimension_order (str): Dimension order. Possible values: "normal" or "transpose".
        rotate (bool): Flag indicating whether to rotate the data.
        x2 (array): 1D array of x coordinates.
        y2 (array): 1D array of y coordinates.

    Returns:
        ndarray or tuple: If `datas` contains only one dataset, returns a 2D numpy array of the data.
                         If `datas` contains two datasets, returns a tuple of two 2D numpy arrays.
    """

    if len(datas[0].dims) == 2:
        data_in = datas[0][variable_names[0]][ttime, :].values
        if rotate:
            data_in2 = datas[1][variable_names[1]][ttime, :].values
            uu, vv = vec_rotate_r2g(50, 15, -90, x2, y2, data_in, data_in2, flag=1)
            print("We are rotating data")
            print(len(x2))
            data_in = uu
            data_in2 = vv
    elif dimension_order == "normal":
        data_in = datas[0][variable_names[0]][ttime, dind, :].values
        if rotate:
            data_in2 = datas[1][variable_names[1]][ttime, dind, :].values
            uu, vv = vec_rotate_r2g(50, 15, -90, x2, y2, data_in, data_in2, flag=1)
            print("We are rotating data")
            print(len(x2))
            data_in = uu
            data_in2 = vv
    elif dimension_order == "transpose":
        data_in = datas[0][variable_names[0]][ttime, :, dind].values
        if rotate:
            data_in2 = datas[1][variable_names[1]][ttime, :, dind].values
            uu, vv = vec_rotate_r2g(50, 15, -90, x2, y2, data_in, data_in2, flag=1)
            print("We are rotating data")
            print(len(x2))
            data_in = uu
            data_in2 = vv
    if len(datas) == 1:
        return data_in
    elif len(datas) == 2:
        return data_in, data_in2
