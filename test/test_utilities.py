import numpy as np
import sys
import os
import matplotlib.tri as mtri
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from fint.fint import (lon_lat_to_cartesian,
                        create_indexes_and_distances,
                        ind_for_depth,
                        get_no_cyclic,
                        mask_triangulation,
                        )
from fint.ut import (compute_face_coords,
                     scalar_r2g,
                     scalar_g2r,
                     vec_rotate_r2g,
                     )

def test_lon_lat_to_cartesian():
    # Test case 1: lon = 0, lat = 0
    lon = 0
    lat = 0
    expected_coordinates = (6371000, 0, 0)
    assert np.allclose(lon_lat_to_cartesian(lon, lat), expected_coordinates,atol=1)

    # Test case 2: lon = 45, lat = 45
    lon = 45
    lat = 45
    expected_coordinates = (3185500, 3185500, 4504977)
    assert np.allclose(lon_lat_to_cartesian(lon, lat), expected_coordinates,atol=1)

    # Test case 3: lon = 180, lat = 90
    lon = 180
    lat = 90
    expected_coordinates = (0, 0, 6371000)
    assert np.allclose(lon_lat_to_cartesian(lon, lat), expected_coordinates,atol=1)

def test_create_indexes_and_distances():
    # Test case 1
    model_lon = np.array([0, 45, 90])
    model_lat = np.array([0, 45, 90])
    lons = np.array([[10, 20], [30, 40]])
    lats = np.array([[5, 15], [25, 35]])
    k = 2
    expected_distances = np.array([[1239965, 5416082],
                                    [2737701, 4008848],
                                    [2582323, 4178855],
                                    [1188371, 5498974]])
    expected_indices = np.array([[0, 1],[0, 1],[1, 0],[1, 0]])
    distances, indices = create_indexes_and_distances(model_lon, model_lat, lons, lats, k=k)
    assert np.allclose(distances, expected_distances,atol=1)
    assert np.allclose(indices, expected_indices)

    # Test case 2
    model_lon = np.array([-180, 0, 180])
    model_lat = np.array([-90, 0, 90])
    lons = np.array([[20, 30, 40], [50, 60, 70]])
    lats = np.array([[-10, 0, 10], [-20, -30, -40]])
    k = 3
    expected_distances = np.array([[ 2460615,  8190399,  9760938],
                                    [ 3297872,  9009954,  9009954],
                                    [ 4465098,  8190399,  9760938],
                                    [ 5669669,  7308510, 10437635],
                                    [ 6371000,  6784365, 11034895],
                                    [ 5385001,  7740161, 11548173]])
    expected_indices = np.array([[1, 0, 2],
                                [1, 2, 0],
                                [1, 2, 0],
                                [1, 0, 2],
                                [0, 1, 2],
                                [0, 1, 2]])
    distances, indices = create_indexes_and_distances(model_lon, model_lat, lons, lats, k=k)
    assert np.allclose(distances, expected_distances,rtol=1e-07,atol=1)
    assert np.allclose(indices, expected_indices)


def test_ind_for_depth():
    # Test case 1: depth = 100, depths_from_file = [0, 50, 100, 150]
    depth = 100
    depths_from_file = [0, 50, 100, 150]
    expected_index = 2 # Expected output
    assert ind_for_depth(depth, depths_from_file) == expected_index

    # Test case 2: depth = -200, depths_from_file = [-100, -200, -300, -400]
    depth = -200
    depths_from_file = [-100, -150, -187.5, -300]
    expected_index = 2 # Expected output
    assert ind_for_depth(depth, depths_from_file) == expected_index


def test_get_no_cyclic():
    # Test case 1
    x2 = np.array([-110, -100, 90, 100, 110, 120, 130, 140, 150, 160])  # x-coordinates of mesh nodes
    elem = np.array([[0, 1, 2], [1, 4, 5], [6, 7, 8], [7, 8, 9]])  # element connectivity array

    expected_result = np.array([2, 3]) # Expected output
    result = get_no_cyclic(x2, elem) # Call the function

    assert np.array_equal(result, expected_result) # Check if the result matches the expected output

    # Test case 2

    x2 = np.array([-100, -250, -80, -70, -60, -50, -40, -30, -20, -10])  # x-coordinates of mesh nodes
    elem = np.array([[0, 1, 2], [5, 2, 3], [2, 3, 4], [3, 4, 5]])  # element connectivity array

    expected_result = np.array([1, 2, 3]) # Expected output
    result = get_no_cyclic(x2, elem)  # Call the function

    assert np.array_equal(result, expected_result) # Check if the result matches the expected output

def test_mask_triangulation():
    # Test case 1
    data_in = np.array([1, 2, 3, 4, 5, 6, 7])
    x = np.array([0, 1, 2, 3, 4, 5, 6])
    y = np.array([0, 1, 2, 3, 4, 5, 6])
    triangles = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    triang2 = mtri.Triangulation(x, y, triangles=triangles)

    elem = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    no_cyclic_elem = np.array([0, 1, 2, 3, 4])

    expected_mask = np.array([False, False, False, False, False]) # Expected output

    result_triang = mask_triangulation(data_in, triang2, elem, no_cyclic_elem)

    assert np.all(result_triang.mask == expected_mask)

def test_compute_face_coords():
    # Test case 1
    x2 = np.array([-110, -100, 90, 100, 110, 120, 130, 140, 150, 160])  # x-coordinates of mesh nodes
    y2 = np.array([45,55,65,75,85,23,43,54,67,79])  # Y coordinates
    elem = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]])  # Element indices

    expected_face_x = np.array([-40, 110, 53]) # Expected output
    expected_face_y = np.array([55, 61, 47])

    face_x, face_y = compute_face_coords(x2, y2, elem) # Call the function

    assert np.allclose(face_x, expected_face_x, atol=1)
    assert np.allclose(face_y, expected_face_y, atol=1)


def test_scalar_g2r():
    # Test case 1
    al = 50
    be = 15
    ga = -90
    lon = np.array([0, 45, 90])
    lat = np.array([0, 30, 60])

    expected_lon = np.array([40.98,  93.74, 144.37]) # Expected output
    expected_lat = np.array([11.44, 30.16, 48.88])

    result_lon, result_lat = scalar_g2r(al, be, ga, lon, lat) # Call the function

    assert np.allclose(result_lon, expected_lon, atol=1e-1)
    assert np.allclose(result_lat, expected_lat, atol=1e-1)


def test_scalar_r2g():
    # Test case 1
    al = 50
    be = 15
    ga = -90
    rlon = np.array([40.98,  93.74, 144.37])
    rlat = np.array([11.44, 30.16, 48.88])

    expected_lon = np.array([0.0, 45.0, 90.0]) # Expected output
    expected_lat = np.array([0.0, 30.0, 60.0])

    result_lon, result_lat = scalar_r2g(al, be, ga, rlon, rlat) # Call the function

    assert np.allclose(result_lon, expected_lon, atol=1e-1)
    assert np.allclose(result_lat, expected_lat, atol=1e-1)


def test_vec_rotate_r2g():
    # Test case 1
    al = 50
    be = 15
    ga = -90
    lon = np.array([0, 45, 90])
    lat = np.array([0, 30, 60])
    urot = np.array([1, 2, 3])
    vrot = np.array([4, 5, 6])
    flag = 0

    expected_u = np.array([1,  0.99, -0.19]) # Expected output
    expected_v = np.array([4, 5.29, 6.70])

    result_u, result_v = vec_rotate_r2g(al, be, ga, lon, lat, urot, vrot, flag) # Call the function

    assert np.allclose(result_u, expected_u, atol=1e-1)
    assert np.allclose(result_v, expected_v, atol=1e-1)

    # Test case 2
    al = 50
    be = 15
    ga = -90
    lon = np.array([0, 45, 90])
    lat = np.array([0, 30, 60])
    urot = np.array([1, 2, 3])
    vrot = np.array([4, 5, 6])
    flag = 1

    expected_u = np.array([0.31, 0.42, 1.05]) # Expected output
    expected_v = np.array([4.11, 5.37, 6.63])

    result_u, result_v = vec_rotate_r2g(al, be, ga, lon, lat, urot, vrot, flag) # Call the function

    assert np.allclose(result_u, expected_u, atol=1e-1)
    assert np.allclose(result_v, expected_v, atol=1e-1)