import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import median_filter
from datetime import datetime
import netCDF4 as nc
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.stats import circmean

# --- Useful stuff ---
cmap_vel = LinearSegmentedColormap.from_list(
    "white_red",
    [(1.0, 1.0, 1.0), (0.8, 0.0, 0.0)],
    N=256
)

# --- Path Setup ---
os.chdir(r"C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\code")
datafolder = r'Z:\glazio\projects\8045-VAW_CAIRN-GLOBAL\SMB_inversions\2024_Argentiere\argentiere_pleiades_smb\data\velocity\PLEIADES_ALPS'
resultsfolder = '../output/velocity/'
os.makedirs(resultsfolder, exist_ok=True)

# --- Load Data (Shapefiles) ---
glc_shp = gpd.read_file('../data/gis/BossonsTaconnaz_rgi6_utm32n.shp')
off_gl_shp = gpd.read_file(r"Z:\glazio\projects\8045-VAW_CAIRN-GLOBAL\SMB_inversions\2024_Argentiere\argentiere_pleiades_smb\data\gis\OffGlacier_mask\velocity_off_glacier_large.shp")

# --- Load & filter velocity data ---
velocity_file = os.path.join(datafolder, 'stack_median_pleiades_alllayers_2012-2022.nc')

# 1. Load data from NetCDF
with nc.Dataset(velocity_file, 'r') as ds:
    v = ds.variables['v'][:]
    vx = ds.variables['vx'][:]
    vy = ds.variables['vy'][:]
    time_data = ds.variables['time'][:]
    
n_y, n_x, n_time  = v.shape
invalid_pairs_index = []

# 2. Filtering based on time interval (NumDays)
filtered_time = []
for ii in range(n_time):
    try:
        t_str = time_data[ii]
        
        dates = t_str.split()
        dt1 = datetime.strptime(dates[0], '%Y-%m-%d')
        dt2 = datetime.strptime(dates[1], '%Y-%m-%d')
        num_days = abs((dt2 - dt1).days)
        
        if num_days > 90:
            if num_days < 320 or num_days > 410:
                invalid_pairs_index.append(ii)
            else:
                filtered_time.append(t_str)
        else:
            filtered_time.append(t_str)
    except (ValueError, IndexError):
        invalid_pairs_index.append(ii)

# Remove invalid time slices
v = np.delete(v, invalid_pairs_index, axis=2)
vx = np.delete(vx, invalid_pairs_index, axis=2)
vy = np.delete(vy, invalid_pairs_index, axis=2)

# 3. Filtering based on velocity direction
# Calculate spatial mean velocity magnitude for the threshold (10 m/yr)
mean_v_spatial = np.nanmean(v, axis=2)

# plot
plt.figure(figsize=(6, 5))
im = plt.imshow(mean_v_spatial, cmap=cmap_vel, vmin=0, vmax=100)
plt.colorbar(im, label="mean velocity (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

# Calculate direction (angles) for all pixels and time steps
angles = np.degrees(np.arctan2(vx, vy))

# Calculate mean direction across the time dimension (axis 2)
# high_low=(low, high) defines the range (e.g., -180 to 180 or 0 to 360)
mean_dir = circmean(angles, high=180, low=-180, axis=2, nan_policy='omit')

plt.figure(figsize=(6, 5))
im = plt.imshow(mean_dir, cmap=cmap_vel, vmin=-180, vmax=180)
plt.colorbar(im, label="velocity angle (°)")
plt.axis("off")
plt.tight_layout()

# Vectorized Angle Difference calculation:
angle_diff = (angles - mean_dir[:, :, np.newaxis] + 180) % 360 - 180

# Create mask: (abs(diff) > 15) AND (mean_velocity > 10)
# We broadcast mean_v_spatial across the time dimension
mask = (np.abs(angle_diff) > 15) & (mean_v_spatial[:, :, np.newaxis] > 10)

# Apply filtering
v[mask] = np.nan
vx[mask] = np.nan
vy[mask] = np.nan

count = v.shape[2]

# reference masks
# 1. Get grid info from NetCDF (assuming X and Y variables exist)
with nc.Dataset(velocity_file, 'r') as ds:
    x_coords = ds.variables['x'][:] # or 'lon'
    y_coords = ds.variables['y'][:] # or 'lat'
    
# 2. Calculate resolution (pixel size) Assumes regular spacing
res_x = x_coords[1] - x_coords[0]
res_y = y_coords[1] - y_coords[0]

# 3. Create the Transform (The "mapping" from pixels to coordinates)
# from_origin(west, north, xsize, ysize)
# Note: NetCDF Y usually goes south-to-north (positive res), 
transform = from_origin(x_coords.min(), y_coords.max(), abs(res_x), abs(res_y))

# 4. Create the mask
# out_shape should match the (height, width) of your 'v' array slices
out_shape = (len(y_coords), len(x_coords))

glc_mask = geometry_mask(
    glc_shp.geometry, 
    transform=transform, 
    invert=True, 
    out_shape=out_shape
)

off_gl_mask = geometry_mask(
    off_gl_shp.geometry, 
    transform=transform, 
    invert=True, 
    out_shape=out_shape
)

plt.imshow(glc_mask)
plt.imshow(off_gl_mask)


# --- Filtering logic & Uncertainty calculation ---
# 1. Extract all off-glacier velocity values across all time slices
vx_off = vx[off_gl_mask,:]
vy_off = vy[off_gl_mask,:]

# 2. Calculate variance for each pair (axis=0 is the pixel axis)
# This gives an array of sigma2 values, one for each time slice
sigma2_per_pair = np.nanstd(vx_off, axis=0)**2 + np.nanstd(vy_off, axis=0)**2

# 3. Handle the "invalid pairs" logic (e.g., if uncertainty > 10m/yr or too many NaNs)
# Logic: Keep slices where sigma2 is not NaN
valid_mask = ~np.isnan(sigma2_per_pair)

# 4. Calculate S and S2 (Aggregates)
S = np.sum(sigma2_per_pair[valid_mask])
# Logic for S2: "assume uncertainty cannot be higher than 10m/yr (100 for variance)"
S2 = np.sum(np.maximum(sigma2_per_pair[valid_mask], 100))
Npairs = np.sum(valid_mask)

unc = np.sqrt(S / Npairs)
unc2 = np.sqrt(S2 / Npairs)

# Remove invalid slices
v_filtered = v[:, :, valid_mask]
vx_filtered = vx[:, :, valid_mask]
vy_filtered = vy[:, :, valid_mask]

# Calculate the final temporal median
v_median = np.nanmedian(v_filtered, axis=2)
vx_median = np.nanmedian(vx_filtered, axis=2)
vy_median = np.nanmedian(vy_filtered, axis=2)

plt.figure(figsize=(6, 5))
im = plt.imshow(v_median, cmap=cmap_vel, vmin=0, vmax=100)
plt.colorbar(im, label="median velocity (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

# --- Spatial Noise Filtering (High pass) ---
# 1. Spatial median filter (11x11 window)
v_spatial_median = median_filter(v_median, size=11)

# 2. Identify noise (where the pixel differs from its neighbors by > 5 m/yr)
res_v = v_median - v_spatial_median
noise_mask = np.abs(res_v) > 5

# 3. Apply the noise mask to the x and y components
vx_median[noise_mask] = np.nan
vy_median[noise_mask] = np.nan

# 4. Final velocity magnitude
v_median_final = np.sqrt(vx_median**2 + vy_median**2)

# --- Saving Results ---
output_meta = {
    'driver': 'GTiff',
    'height': out_shape[0],
    'width': out_shape[1],
    'count': 1,
    'dtype': 'float32',
    'crs': 'EPSG:32632', 
    'transform': transform,
    'nodata': np.nan
}

# Save the final median velocity
with rasterio.open(os.path.join(resultsfolder, 'v_median_after_30_15.tif'), 'w', **output_meta) as dst:
    dst.write(v_median_final.astype(np.float32), 1)

# Save components
for name, data in [('vx_median', vx_median), ('vy_median', vy_median)]:
    with rasterio.open(os.path.join(resultsfolder, f'{name}_after_30_15.tif'), 'w', **output_meta) as dst:
        dst.write(data.astype(np.float32), 1)
