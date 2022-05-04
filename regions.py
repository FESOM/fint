import xarray as xr
import numpy as np
import matplotlib.pylab as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print(
        "Cartopy is not installed, interpolation to projected regions is not available."
    )
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
try:
    import shapely.vectorized
except ImportError:
    print(
        "Shapely is not installed, use --no_shape_mask to make things work with nearest neighbour interpolation."
    )

def define_region_from_file(file):
    data_region = xr.open_dataset(file)
    if ("lon" not in data_region.data_vars) or ("lat" not in data_region.data_vars):
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
    if not projection is None:
        x, y, lon, lat = region_cartopy(box, res, projection)
    # box_type = parse_box(box)
    elif len(box.split(",")) > 1:
        left, right, down, up = list(map(float, box.split(",")))
        if not res is None:
            lonNumber, latNumber = res
        else:
            lonNumber, latNumber = 360, 170

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
    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 300, 250
    left = -80
    right = -30
    bottom = 20
    top = 60
    x = np.linspace(left, right, lonNumber)
    y = np.linspace(bottom, top, latNumber)
    lon, lat = np.meshgrid(x, y)

    return x, y, lon, lat


def region_trop(res):
    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 751, 400
    left = -60
    right = 15
    bottom = -9.95
    top = 29.95
    x = np.linspace(left, right, lonNumber)
    y = np.linspace(bottom, top, latNumber)
    lon, lat = np.meshgrid(x, y)

    return x, y, lon, lat


def region_arctic(res):
    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 500, 500
    left = -180
    right = 180
    bottom = 60
    top = 90
    x, y, lon, lat = region_cartopy(
        f"{left},{right},{bottom},{top}", res, projection="np"
    )

    return x, y, lon, lat


def region_gulf(res):
    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 1000, 500
    left = -80
    right = -30
    bottom = 20
    top = 60
    x, y, lon, lat = region_cartopy(
        f"{left},{right},{bottom},{top}", res, projection="mer"
    )

    return x, y, lon, lat


def region_cartopy(box, res, projection="mer"):
    if projection == "mer":
        projection_ccrs = ccrs.Mercator()
    elif projection == "np":
        projection_ccrs = ccrs.NorthPolarStereo()

    if not res is None:
        lonNumber, latNumber = res
    else:
        lonNumber, latNumber = 500, 500
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
    """Mask earth from lon/lat data using Natural Earth.
    Parameters
    ----------
    lonreg2: float, np.array
        2D array of longitudes
    latreg2: float, np.array
        2D array of latitudes
    Returns
    -------
    m2: bool, np.array
        2D mask with True where the ocean is.
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
