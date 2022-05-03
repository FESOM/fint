import os
import numpy as np


def update_attrs(ds, args):
    if args.target is not None:
        ds["target"] = args.target

    ds["box"] = args.box
    ds["influence"] = args.influence
    ds["interp"] = args.interp
    ds["data"] = os.path.abspath(args.data)
    ds["meshpath"] = os.path.abspath(args.meshpath)
    return ds

def nodes_or_ements(data, variable_name, node_num, elem_num):
    if data[variable_name].shape[-1] == node_num:
        return "nodes"
    elif data[variable_name].shape[-1] == elem_num:
        return "elements"

def compute_face_coords(x2, y2, elem):
    """ Compute coordinates of elements (triangles)
    Parameters:
    -----------
    mesh: mesh object
        fesom mesh object
    Returns:
    --------
    face_x: numpy array
        x coordinates
    face_y: numpy array
        y coordinates
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
    vector_vars = {}
    vector_vars["u"] = ["u", "v"]
    vector_vars["v"] = ["u", "v"]
    vector_vars["uice"] = ["uice", "vice"]
    vector_vars["vice"] = ["uice", "vice"]

    return vector_vars[variable_name]

