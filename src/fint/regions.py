import matplotlib.pylab as plt
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print(
        "Cartopy is not installed, interpolation to projected regions is not available."
    )
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage, TextArea

try:
    import shapely.vectorized
except ImportError:
    print(
        "Shapely is not installed, use --no_shape_mask to make things work with nearest neighbour interpolation."
    )
from .ut import convert_lon_lat_to_180, convert_lon_lat_0_360


def define_region_from_file(file):
    """
    Defines region coordinates from a file.

    Parameters:
        file (str): Path to the file containing region data.

    Returns:
        - ndarray: 1D array of x coordinates.
        - ndarray: 1D array of y coordinates.
        - ndarray: 2D array of longitude values.
        - ndarray: 2D array of latitude values.

    Raises:
        ValueError: If the file does not contain lon or lat coordinates.
    """
    data_region = xr.open_dataset(file)
    if (
        ("lat" not in data_region.coords)
        and ("lat" not in data_region.data_vars)
        and ("lon" not in data_region.coords)
        and ("lon" not in data_region.data_vars)
    ):
        raise ValueError("No lon or lat in target file")

    lon = data_region.lon.values
    lat = data_region.lat.values

    if len(lon.shape) == 1:
        lon, lat = np.meshgrid(lon, lat)
    elif len(lon.shape) == 2:
        lon, lat = lon, lat
    elif len(lon.shape) == 3:
        lon, lat = lon[0, :], lat[0, :]

    x = np.linspace(lon.min(), lon.max(), lon.shape[1])
    y = np.linspace(lat.min(), lat.max(), lat.shape[0])
    return x, y, lon, lat


def define_region(box, res, projection=None):
    """
    Defines region coordinates based on the specified box and resolution.

    Parameters:
        box (str): Specifies the region of interest. It can be one of the following options:
                   - A comma-separated string specifying the bounding box coordinates: left, right, down, up.
                   - "gs" for the global-scale region.
                   - "trop" for the tropical region.
                   - "arctic" for the Arctic region.
                   - "gulf" for the Gulf region.
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(360, 170) would generate a grid with 360 longitude points and 170 latitude points.
        projection (optional): Specifies the projection to use when generating the coordinates.
                               If None, the coordinates will be in a regular lat/lon grid.

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.

    Raises:
        ValueError: If an unknown or unsupported region is specified.
    """

    if not projection is None:
        x, y, lon, lat = region_cartopy(box, res, projection)
    # box_type = parse_box(box)
    elif len(box.split(",")) > 1:
        left, right, down, up = list(map(float, box.split(",")))
        if not res is None:
            lonNumber, latNumber = res
        else:
            res = (360,170)
            lonNumber, latNumber = res

        x = np.linspace(left, right, lonNumber)
        y = np.linspace(down, up, latNumber)

        lon, lat = np.meshgrid(x, y)
    elif box == "gs":
        x, y, lon, lat = region_gs(res)
    elif box == "trop":
        x, y, lon, lat = region_trop(res)
    elif box == "arctic":
        x, y, lon, lat = region_arctic(res)
    elif box == "gulf":
        x, y, lon, lat = region_gulf(res)
    else:
        raise ValueError("Unknown region")

    return x, y, lon, lat


def region_gs(res):
    """
    Generate coordinates for the global-scale region.

    Parameters:
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(300, 250) would generate a grid with 300 longitude points and 250 latitude points.

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.
    """
    if not res is None:
        lonNumber, latNumber = res
    else:
        res = (300,250)
        lonNumber, latNumber = res
    left = -80
    right = -30
    bottom = 20
    top = 60
    x = np.linspace(left, right, lonNumber)
    y = np.linspace(bottom, top, latNumber)
    lon, lat = np.meshgrid(x, y)

    return x, y, lon, lat


def region_trop(res):
    """
    Generate coordinates for the tropical region.

    Parameters:
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(751, 400) would generate a grid with 751 longitude points and 400 latitude points.

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.
    """
    if not res is None:
        lonNumber, latNumber = res
    else:
        res = (751,400)
        lonNumber, latNumber = res
    left = -60
    right = 15
    bottom = -9.95
    top = 29.95
    x = np.linspace(left, right, lonNumber)
    y = np.linspace(bottom, top, latNumber)
    lon, lat = np.meshgrid(x, y)

    return x, y, lon, lat


def region_arctic(res):
    """
    Generate coordinates for the Arctic region.

    Parameters:
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(500, 500) would generate a grid with 500 longitude points and 500 latitude points.

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.
    Note:
        - The generated coordinates cover the Arctic region, spanning from -180° to 180° longitude and from 60° to 90° latitude.
        - The projection used is the NorthPolarStereo projection
    """
    if not res is None:
        lonNumber, latNumber = res
    else:
        res = (500,500)
        lonNumber, latNumber = res
    left = -180
    right = 180
    bottom = 60
    top = 90
    x, y, lon, lat = region_cartopy(
        f"{left},{right},{bottom},{top}", res, projection="np"
    )

    return x, y, lon, lat


def region_gulf(res):
    """
    Generate coordinates for the Gulf region.

    Parameters:
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(1000, 500) would generate a grid with 1000 longitude points and 500 latitude points.

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.

    Note:
        - The generated coordinates cover the Gulf region, spanning from -80° to -30° longitude and from 20° to 60° latitude.
        - The projection used is the Mercator projection.
    """
    if not res is None:
        lonNumber, latNumber = res
    else:
        res = (1000, 500)
        lonNumber, latNumber = res
    left = -80
    right = -30
    bottom = 20
    top = 60
    x, y, lon, lat = region_cartopy(
        f"{left},{right},{bottom},{top}", res, projection="mer"
    )

    return x, y, lon, lat


def region_cartopy(box, res, projection="mer"):
    """
    Generate coordinates using Cartopy projections.

    Parameters:
        box (str): A string defining the bounding box for the desired region.
                   The box string should be in the format "left,right,bottom,top",
                   where "left" is the minimum longitude, "right" is the maximum longitude,
                   "bottom" is the minimum latitude, and "top" is the maximum latitude.
        res (tuple): A tuple containing the desired number of longitude and latitude grid points.
                     For example, res=(500, 500) would generate a grid with 500 longitude points and 500 latitude points.
        projection (str, optional): The desired Cartopy projection to use.
                                    Options are "mer" (Mercator), "np" (NorthPolarStereo), and "sp" (SouthPolarStereo).
                                    Default is "mer" (Mercator).

    Returns:
        tuple containing

        - x (ndarray): 1D array of coordinates.
        - y (ndarray): 1D array of coordinates.
        - lon (ndarray): 2D array of longitude values.
        - lat (ndarray): 2D array of latitude values.
    """
    if projection == "mer":
        projection_ccrs = ccrs.Mercator()
    elif projection == "np":
        projection_ccrs = ccrs.NorthPolarStereo()
    elif projection == "sp":
        projection_ccrs = ccrs.SouthPolarStereo()

    if not res is None:
        lonNumber, latNumber = res
    else:
        res = (500,500)
        lonNumber, latNumber = res
    left, right, down, up = list(map(float, box.split(",")))
    print(left, right, down, up)
    # left = -80
    # right = -30
    # bottom = 20
    # top = 60
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw=dict(projection=projection_ccrs),
        constrained_layout=True,
        figsize=(10, 10),
    )
    sstep = 10
    ax.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
    # ax.coastlines(resolution = '110m',lw=0.5)

    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    # res = scl_fac * 300. # last number is the grid resolution in meters (NEEDS TO BE CHANGED)
    # nx = int((xmax-xmin)/res)+1; ny = int((ymax-ymin)/res)+1
    x = np.linspace(xmin, xmax, lonNumber)
    y = np.linspace(ymin, ymax, latNumber)
    x2d, y2d = np.meshgrid(x, y)
    print(x.shape)

    npstere = ccrs.PlateCarree()
    # transformed2 =  npstere.transform_points(ccrs.NorthPolarStereo(), x, y)
    transformed2 = npstere.transform_points(projection_ccrs, x2d, y2d)
    lon = transformed2[:, :, 0]  # .ravel()
    lat = transformed2[:, :, 1]  # .ravel()
    print(lon.shape)

    # left = -60
    # right = 15
    # bottom = -9.95
    # top = 29.95
    # x = np.linspace(left,right,lonNumber)
    # y = np.linspace(bottom,top,latNumber)
    # lon, lat = np.meshgrid(x,y)

    return x, y, lon, lat


def mask_ne(lonreg2, latreg2):
    """
    Masks the Earth from lon/lat data using Natural Earth.

    Parameters:
        lonreg2 (np.array): 2D array of longitudes.
        latreg2 (np.array): 2D array of latitudes.

    Returns:
        m2 (np.array): 2D mask with True where the ocean is.
    """
    nearth = cfeature.NaturalEarthFeature("physical", "ocean", "50m")
    main_geom = [contour for contour in nearth.geometries()][0]

    mask = shapely.vectorized.contains(main_geom, lonreg2, latreg2)
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == -180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 > 71.5)), True, mask)
    m2 = np.where(
        ((lonreg2 == 180.0) & (latreg2 < 70.95) & (latreg2 > 68.96)), True, m2
    )
    m2 = np.where(((lonreg2 == -180.0) & (latreg2 < 65.33)), True, m2)
    m2 = np.where(((lonreg2 == 180.0) & (latreg2 < 65.33)), True, m2)

    return ~m2
