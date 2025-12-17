import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio import features

# --- Initialize Paths ---
base_path = r'C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\code'
os.chdir(base_path)

# --- Resampling function ---
def resample_and_clip(source_ds, target_res_ds, target_bounds):
    """
    Resamples source_ds to match target_res_ds resolution, 
    but crops the result to target_bounds.
    """
    # Reproject to match the resolution and CRS of the thickness data
    # 'bilinear' is used for continuous data (velocity/elevation)
    resampled = source_ds.rio.reproject_match(
        target_res_ds, 
        resampling=Resampling.average # here we're downsampling so using a method that ignores NaNs
    )
    
    # Clip to the DEM bounding box
    # *dem_bounds unpacks (minx, miny, maxx, maxy)
    clipped = resampled.rio.clip_box(*target_bounds)
    
    return clipped

# --- Load Data ---
# Load shapefile
glacier_shp = gpd.read_file('../data/gis/BossonsTaconnaz_rgi6_utm32n.shp')

# Load Rasters using rioxarray
dem_ds = rioxarray.open_rasterio(r'..\output\dh_results\meanDEM-2017_02_15.tif').squeeze()
vx_ds = rioxarray.open_rasterio(r'..\output\velocity\vx_median_after_30_15.tif').squeeze()
vy_ds = rioxarray.open_rasterio(r'..\output\velocity\vy_median_after_30_15.tif').squeeze()
thx_ds = rioxarray.open_rasterio(r'..\data\thx\Millan_thx_utm32n.tif').squeeze()

plt.imshow(dem_ds)
plt.imshow(vx_ds)
plt.imshow(thx_ds)

# Load GPR Points (Shapefile with attributes) - to be added if there are any
#thx_obs_gdf = gpd.read_file(r'..\output\thx\GPR_points_Rhone_shifted.shp')

# Process GPR Points
# Filter out negative thickness
#thx_obs_gdf = thx_obs_gdf[thx_obs_gdf['thk'] >= 0]
#thx_obs_thx = thx_obs_gdf['thk'].values
#thx_obs_x = thx_obs_gdf.geometry.x.values
#thx_obs_y = thx_obs_gdf.geometry.y.values



# --- Resampling to Thickness Resolution (thx) & DEM bounds ---

# 1. Get the target bounds from the DEM
# This returns (west, south, east, north)
#dem_bounds = dem_ds.rio.bounds()
dem_bounds = (331000, 5076000, 337000, 5085000)

# 2. Apply to datasets
vx_final = resample_and_clip(vx_ds, thx_ds, dem_bounds)
vy_final = resample_and_clip(vy_ds, thx_ds, dem_bounds)

plt.imshow(vx_final)
plt.imshow(vy_final)

# If the thickness data itself is larger than the DEM, clip it too:
thx_final = thx_ds.rio.clip_box(*dem_bounds)
plt.imshow(thx_final)

# Clean main thickness raster
thx_data = thx_final.values
thx_data[thx_data < 0] = 0

plt.imshow(thx_data)

dem_final = resample_and_clip(dem_ds, thx_ds, dem_bounds)
plt.imshow(dem_final)

# Rasterize Glacier Mask (1 is inside the glacier shape and 0 is outside)
glacier_mask = features.rasterize(
    [(shape, 1) for shape in glacier_shp.geometry],
    out_shape=thx_final.shape,
    transform=thx_final.rio.transform(),
    fill=0,
    all_touched=True
)
plt.imshow(glacier_mask)

# --- Binning Point Measurements (Grid Loop) - if there ---
# Note: Could be sped up with Pandas/NumPy
xt = thx_final.x.values
yt = thx_final.y.values
#thx_obs_grid = np.full((len(yt), len(xt)), np.nan)
#
## Pre-calculate cell boundaries for speed
#for i in range(len(yt) - 1):
#    for j in range(len(xt) - 1):
#        # Determine bounds (handling potentially descending Y coordinates)
#        x_min, x_max = xt[j], xt[j+1]
#        y_min, y_max = min(yt[i], yt[i+1]), max(yt[i], yt[i+1])
#        
#        # Mask points within this cell
#        mask = (thx_obs_x >= x_min) & (thx_obs_x < x_max) & \
#               (thx_obs_y >= y_min) & (thx_obs_y < y_max)
#        
#        if np.any(mask):
#            thx_obs_grid[i, j] = np.mean(thx_obs_thx[mask])

# --- Prepare Final Variables ---
ZZ = np.nan_to_num(dem_final, nan=0.0)
THKINIT = thx_data.copy()
THKINIT[glacier_mask == 0] = 0

ICEMASKOBS = glacier_mask
UVELSURFOBS = np.where(ICEMASKOBS == 1, vx_final, np.nan)
VVELSURFOBS = np.where(ICEMASKOBS == 1, vy_final, np.nan)
#THKOBS = thx_obs_grid
# if no thickness observations, just empty raster
THKOBS = THKINIT*np.nan

# --- Save to GeoTIFF ---
output_dir = '../igm/run_2025_12_17/'
os.makedirs(output_dir, exist_ok=True)

def save_tiff(data, name, reference_ds):
    da = xr.DataArray(data, coords=reference_ds.coords, dims=reference_ds.dims)
    da.rio.write_crs(reference_ds.rio.crs, inplace=True)
    da.rio.to_raster(os.path.join(output_dir, name))

save_tiff(ZZ, 'usurf.tif', thx_final)
save_tiff(ZZ, 'usurfobs.tif', thx_final)
save_tiff(THKINIT, 'thkinit.tif', thx_final)
save_tiff(THKOBS, 'thkobs.tif', thx_final)
save_tiff(ICEMASKOBS, 'icemaskobs.tif', thx_final)
save_tiff(ICEMASKOBS, 'icemask.tif', thx_final)
save_tiff(UVELSURFOBS, 'uvelsurfobs.tif', thx_final)
save_tiff(VVELSURFOBS, 'vvelsurfobs.tif', thx_final)

print("Processing Complete.")