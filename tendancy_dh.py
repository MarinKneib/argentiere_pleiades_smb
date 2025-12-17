# This script extracts the trends from multiple (already co-registered) DEMs

# Initialize
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import xy
from scipy.interpolate import griddata
from scipy.ndimage import median_filter
from skimage.morphology import remove_small_objects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from rasterio.windows import from_bounds
from rasterio.windows import Window

# working directory
os.chdir(r"C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\code")

# folders
datafolder = r"C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\data\DEMs"
resultsfolder = "../output/dh_results/"
os.makedirs(resultsfolder, exist_ok=True)

# DEM dates
dates = [
    "20210815","20200930","20200917","20200809","20190825",
    "20180908","20180812","20170814","20160928","20160807",
    "20150830","20130920","20120819"
]

date_t = np.array([datetime.strptime(d, "%Y%m%d") for d in dates])
time = np.array([dt.timestamp() for dt in date_t])

# shapefiles
glc_shp = gpd.read_file("../data/gis/BossonsTaconnaz_rgi6_utm32n.shp")
OffGl = gpd.read_file(
    r"Z:\glazio\projects\8045-VAW_CAIRN-GLOBAL\SMB_inversions\2024_Argentiere\argentiere_pleiades_smb\data\gis\OffGlacier_mask\velocity_off_glacier_large.shp"
)

# Load reference DEM
ref_path = os.path.join(datafolder, "20150830_DEM_4m_shift_H-V_clip.tif")

# Reference DEM bounds in map coordinates
#ref_bounds = rasterio.transform.array_bounds(
#    refDEM.shape[0],
#    refDEM.shape[1],
#    transform
#)
ref_bounds = (331000, 5076000, 344000, 5091000) # (xmin, ymin, xmax, ymax)

with rasterio.open(ref_path) as src:
    # Compute window
    win = from_bounds(*ref_bounds, transform=src.transform)

    # Read clipped DEM
    refDEM = src.read(1, window=win).astype(float)

    # Update transform for clipped DEM
    transform = src.window_transform(win)

    crs = src.crs
    profile = src.profile.copy()

# Apply altitude threshold
refDEM[refDEM < 500] = np.nan

# define reference grid
ny, nx = refDEM.shape
rows, cols = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
Xdem, Ydem = rasterio.transform.xy(transform, rows, cols)
Xdem = np.array(Xdem)
Ydem = np.array(Ydem)

# plot DEM
plt.imshow(refDEM)

# Load all DEMs
DEM3d = np.full((ny, nx, len(dates)), np.nan, dtype=np.float32) # pre-allocate array full of NaNs

for i, d in enumerate(dates):
    print(f'loading {d} DEM')
    path = os.path.join(datafolder, f"{d}_DEM_4m_shift_H-V_clip.tif")
    with rasterio.open(path) as src:

        # Compute window corresponding to reference bounds
        win = from_bounds(*ref_bounds, transform=src.transform)

        # Round safely to integer window
        win = win.round_offsets().round_lengths()

        # Clip window to DEM extent
        dem_win = win.intersection(Window(0, 0, src.width, src.height))

        if dem_win.width == 0 or dem_win.height == 0:
            # No overlap at all → leave as NaNs
            continue

        # Read only that window
        DEMt = src.read(1, window=dem_win).astype(np.float32)
        DEMt[DEMt < 500] = np.nan # NaN if elevation lower than 500 m (should not happen in this zone)

        dem_transform = src.window_transform(dem_win)

    # Compute where this window falls in the reference grid
    ref_win = from_bounds(
        *rasterio.transform.array_bounds(
            DEMt.shape[0], DEMt.shape[1], dem_transform
        ),
        transform=transform
    )

    ref_win = ref_win.round_offsets().round_lengths()

    # Intersection safety (numerical precision)
    ref_win = ref_win.intersection(Window(0, 0, nx, ny))

    if ref_win.width == 0 or ref_win.height == 0:
        continue

    # Compute slicing indices
    r0 = int(ref_win.row_off)
    c0 = int(ref_win.col_off)
    r1 = r0 + int(ref_win.height)
    c1 = c0 + int(ref_win.width)

    DEM3d[r0:r1, c0:c1, i] = DEMt[:ref_win.height, :ref_win.width]

#### dh/dt regression (could still be made faster with numba)
tendancy = np.full((ny, nx), np.nan)
r2 = np.full((ny, nx), np.nan)

# pre-compile valid counts & time span
valid_mask = ~np.isnan(DEM3d)

valid_count = valid_mask.sum(axis=2)

time_span = np.nanmax(
    np.where(valid_mask, time, np.nan), axis=2
) - np.nanmin(
    np.where(valid_mask, time, np.nan), axis=2
)

eligible = (valid_count >= len(time) - 5) & (time_span >= 5 * 365 * 24 * 3600)

plt.imshow(eligible)

# loop through eligible pixels
idxs = np.argwhere(eligible)

for i, j in idxs:
    pix = DEM3d[i, j, :]
    valid = ~np.isnan(pix) # remove Nans in time series

    t_valid = time[valid]
    z_valid = pix[valid]
        
    # regression using analytical formula
    t_mean = t_valid.mean()
    z_mean = z_valid.mean()

    cov = np.mean((t_valid - t_mean) * (z_valid - z_mean))
    var = np.mean((t_valid - t_mean) ** 2)
    
    b1 = cov / var
    b0 = z_mean - b1 * t_mean

    ycalc = b0 + b1 * t_valid

    resid = z_valid - ycalc
    good = np.abs(resid) <= 3 * np.std(resid)

    # second pass
    t2 = t_valid[good]
    z2 = z_valid[good]

    t_mean = t2.mean()
    z_mean = z2.mean()

    b1 = np.mean((t2 - t_mean) * (z2 - z_mean)) / np.mean((t2 - t_mean) ** 2)
    b0 = z_mean - b1 * t_mean

    ycalc = b0 + b1 * t2

    tendancy[i, j] = b1
    r2[i, j] = 1 - np.sum((z2 - ycalc) ** 2) / np.sum((z2 - z2.mean()) ** 2)

# convert to m/yr
tendancy *= 3600 * 24 * 365
tendancy[(tendancy < -10) | (tendancy > 5)] = np.nan # remove unrealistic values

### plot
# Red – White – Blue (negative → positive)
cmap_tend = LinearSegmentedColormap.from_list(
    "rwb",
    [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.8)],
    N=256
)
cmap_slope = LinearSegmentedColormap.from_list(
    "white_red",
    [(1.0, 1.0, 1.0), (0.8, 0.0, 0.0)],
    N=256
)

plt.figure(figsize=(6, 5))
im = plt.imshow(tendancy, cmap=cmap_tend, vmin=-10, vmax=10)
plt.colorbar(im, label="dh/dt (m yr⁻¹)")
plt.title("Elevation change rate (tendancy)")
plt.axis("off")
plt.tight_layout()

### Mask glacier
glc_mask = geometry_mask(
    glc_shp.geometry,
    out_shape=refDEM.shape,
    transform=transform,
    invert=True
).astype(float)

# slope
dx = abs(transform.a)   # pixel width 
dy = abs(transform.e)   # pixel height 

def slope_horn(dem, dx, dy):
    """
    Approximate TopoToolbox gradient8 using Horn (1981) method
    """
    z = dem

    dzdx = (
        (z[:-2, 2:] + 2*z[1:-1, 2:] + z[2:, 2:]) -
        (z[:-2, :-2] + 2*z[1:-1, :-2] + z[2:, :-2])
    ) / (8 * dx)

    dzdy = (
        (z[2:, :-2] + 2*z[2:, 1:-1] + z[2:, 2:]) -
        (z[:-2, :-2] + 2*z[:-2, 1:-1] + z[:-2, 2:])
    ) / (8 * dy)

    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    # pad to original shape
    slope_full = np.full(z.shape, np.nan)
    slope_full[1:-1, 1:-1] = slope

    return slope_full

avgDEM = np.nanmean(DEM3d, axis=2)
slope = slope_horn(avgDEM, dx, dy)

plt.figure(figsize=(6, 5))
im = plt.imshow(slope, cmap=cmap_slope, vmin=0, vmax=50)
plt.colorbar(im, label="Slope (degrees)")
plt.title("Surface slope")
plt.axis("off")
plt.tight_layout()

slope_mask = slope <= 40
slope_mask = remove_small_objects(slope_mask, 10)

# Gap filling
tendancy_gl = tendancy * glc_mask
#tofill = np.isnan(tendancy_gl) & (glc_mask == 1)
#
#plt.figure(figsize=(6, 5))
#im = plt.imshow(tendancy_gl, cmap=cmap_tend, vmin=-10, vmax=10)
#plt.colorbar(im, label="dh/dt (m yr⁻¹)")
#plt.title("Elevation change rate (tendancy)")
#plt.axis("off")
#plt.tight_layout()
#
## Ensure Xdem, Ydem are 2D grids
#if Xdem.ndim == 1 or Ydem.ndim == 1:
#    rows, cols = np.meshgrid(
#        np.arange(tendancy_gl.shape[0]),
#        np.arange(tendancy_gl.shape[1]),
#        indexing="ij"
#    )
#    Xdem, Ydem = rasterio.transform.xy(transform, rows, cols)
#    Xdem = np.asarray(Xdem)
#    Ydem = np.asarray(Ydem)
#
#tendancy_med = median_filter(tendancy_gl, size=3)
#tendancy_gl[tofill] = griddata(
#    (Xdem[~np.isnan(tendancy_med)], Ydem[~np.isnan(tendancy_med)]),
#    tendancy_med[~np.isnan(tendancy_med)],
#    (Xdem[tofill], Ydem[tofill]),
#    method="cubic"
#)

tendancy_gl = median_filter(tendancy_gl, size=3)

tendancy_filt = tendancy.copy()
fill_cond = np.isnan(tendancy) & (glc_mask == 1)
tendancy_filt[fill_cond] = tendancy_gl[fill_cond]

plt.figure(figsize=(6, 5))
im = plt.imshow(tendancy_filt, cmap=cmap_tend, vmin=-10, vmax=10)
plt.colorbar(im, label="dh/dt (m yr⁻¹)")
plt.title("Elevation change rate (tendancy)")
plt.axis("off")
plt.tight_layout()

# Uncertainty
offgl_mask = geometry_mask(
    OffGl.geometry,
    out_shape=refDEM.shape,
    transform=transform,
    invert=True
).astype(float)

offgl_mask = (offgl_mask == 1)

tendancy_offglacier = tendancy_filt.copy()
tendancy_offglacier[~offgl_mask] = np.nan

tendancy_offglacier[slope > 40] = np.nan

print("Off-glacier mean:", np.nanmean(tendancy_offglacier))
print("Off-glacier std:", np.nanstd(tendancy_offglacier))

# glacier mask: 1 = glacier, 0 = outside
glc_mask = (glc_mask == 1)

tendancy_glacier = tendancy_filt.copy()
tendancy_glacier[~glc_mask] = np.nan

print("Glacier mean dh/dt:", np.nanmean(tendancy_glacier))

# Outputs
profile.update(
    height=refDEM.shape[0],
    width=refDEM.shape[1],
    transform=transform,
    dtype="float32",
    nodata=np.nan
)

with rasterio.open(os.path.join(resultsfolder, "tendancy.tif"), "w", **profile) as dst:
    dst.write(tendancy.astype("float32"), 1)

with rasterio.open(os.path.join(resultsfolder, "r2.tif"), "w", **profile) as dst:
    dst.write(r2.astype("float32"), 1)

with rasterio.open(os.path.join(resultsfolder, "tendancy_filt.tif"), "w", **profile) as dst:
    dst.write(tendancy_filt.astype("float32"), 1)

# Extrapolated DEM
refDEM2 = DEM3d[:, :, 3]  # 2020-08-09
dt_years = (datetime(2020, 8, 9) - datetime(2017, 2, 15)).days / 365
medDEM = refDEM2 - tendancy_filt * dt_years

#tofill = np.isnan(medDEM)
#medDEM[tofill] = griddata(
#    (Xdem[~tofill], Ydem[~tofill]),
#    medDEM[~tofill],
#    (Xdem[tofill], Ydem[tofill]),
#    method="cubic"
#)

with rasterio.open(os.path.join(resultsfolder, "meanDEM-2017_02_15.tif"), "w", **profile) as dst:
    dst.write(medDEM.astype("float32"), 1)
