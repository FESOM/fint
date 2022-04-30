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


def define_region(box, res):

    # box_type = parse_box(box)
    if len(box.split(",")) > 1:
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
