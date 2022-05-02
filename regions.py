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
    x = np.linspace(left,right,lonNumber)
    y = np.linspace(bottom,top,latNumber)
    lon, lat = np.meshgrid(x,y)
    
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
    x = np.linspace(left,right,lonNumber)
    y = np.linspace(bottom,top,latNumber)
    lon, lat = np.meshgrid(x,y)
    
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
    x, y, lon, lat = region_cartopy(f"{left},{right},{bottom},{top}", res, projection="np")
    
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
    x, y, lon, lat = region_cartopy(f"{left},{right},{bottom},{top}", res, projection="mer")
    
    return x, y, lon, lat

def region_cartopy(box, res, projection='mer'):
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
                figsize=(10,10),
            )
    sstep = 10
    ax.set_extent([left, right, down, up], crs=ccrs.PlateCarree())
    # ax.coastlines(resolution = '110m',lw=0.5)

    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()

    # res = scl_fac * 300. # last number is the grid resolution in meters (NEEDS TO BE CHANGED)
    # nx = int((xmax-xmin)/res)+1; ny = int((ymax-ymin)/res)+1
    x = np.linspace(xmin,xmax,lonNumber)
    y = np.linspace(ymin,ymax,latNumber)
    x2d, y2d = np.meshgrid(x,y)
    print(x.shape)

    npstere = ccrs.PlateCarree()
    # transformed2 =  npstere.transform_points(ccrs.NorthPolarStereo(), x, y)
    transformed2 =  npstere.transform_points(projection_ccrs, x2d, y2d)
    lon = transformed2[:,:,0]#.ravel()
    lat = transformed2[:,:,1]#.ravel()
    print(lon.shape)

    # left = -60
    # right = 15
    # bottom = -9.95
    # top = 29.95
    # x = np.linspace(left,right,lonNumber)
    # y = np.linspace(bottom,top,latNumber)
    # lon, lat = np.meshgrid(x,y)
    
    return x, y, lon, lat
