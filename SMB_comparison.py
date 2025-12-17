import os
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from scipy import stats
from rasterio import features
from rasterio.enums import Resampling

# --- Setup Paths ---
base_dir = r'C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\MerDeGlace\code'
os.chdir(base_dir)

datafolder1 = '../output/smb_glacioclim/'
datafolder2 = '../output/smb/'
results_dir = '../output/smb_comparison/2025_12_17/'
os.makedirs(results_dir, exist_ok=True)

best = 0

# --- 1. Load Rasters & Resample ---
# We use rioxarray to handle the interpolation (interp2 equivalent)
dem_ds = rioxarray.open_rasterio(r'..\output\dh_results\meanDEM-2017_02_15.tif').squeeze()
dh_ds = rioxarray.open_rasterio('../output/dh_results/tendancy_filt.tif').squeeze()

if best == 0:
    smb_path = f"{datafolder2}2024-12-20_dhHres_fluxIGM_superzack_uncertainty_N=100/MerDeGlace_SMB.tif"
    smbu_path = f"{datafolder2}2024-12-20_dhHres_fluxIGM_superzack_uncertainty_N=100/MerDeGlace_SMBu.tif"
else:
    smb_path = f"{datafolder2}2024-04-17_dhPleiades_fluxIGM_sens_uncertainty_N=1000/Argentiere_SMB_10percbest.tif"
    smbu_path = f"{datafolder2}2024-04-17_dhPleiades_fluxIGM_sens_uncertainty_N=1000/Argentiere_SMBu_10percbest.tif"

smb_calc = rioxarray.open_rasterio(smb_path).squeeze()
smbu_calc = rioxarray.open_rasterio(smbu_path).squeeze()

# Resample DEM and dh to match the SMB grid (Target Grid)
dem_r = dem_ds.rio.reproject_match(smb_calc, resampling=Resampling.average).values
dh_r = dh_ds.rio.reproject_match(smb_calc, resampling=Resampling.average).values

# --- 2. Load Stake Data (CSVs) ---
def load_stake_csv(name):
    path = f"..\output\smb_glacioclim\{name}"
    # Python's pandas handles columns automatically
    df = pd.read_csv(path)
    return df

smb_abl = load_stake_csv("SMB_ABL_2012-2021_utm32n.csv")
smb_acc = load_stake_csv("SMB_ACC_2012-2021_utm32N.csv")
smb_les = load_stake_csv("SMB_LES_2012-2021_utm32N.csv")
smb_tac = load_stake_csv("SMB_TAC_2012-2021_utm32N.csv")

smb_all = pd.concat([smb_abl, smb_acc, smb_les, smb_tac], ignore_index=True)

# --- 3. Extract Raster Values at Stakes ---
# Replaces the mink/nanmean logic with xarray's advanced indexing
def extract_at_points(raster, df):
    # Select nearest points with a 3x3 window (approx) using slice
    vals = []
    for _, row in df.iterrows():
        # Get a small 3x3 window around the point
        window = raster.sel(x=row.X, y=row.Y, method="nearest", tolerance=50) 
        # Note: MATLAB used 3 nearest pixels, here we take the single nearest or mean of small slice
        vals.append(float(window.mean()))
    return np.array(vals)

smb_all['SMB_calc_IGMf'] = extract_at_points(smb_calc, smb_all)
smb_all['SMBu_calc_IGMf'] = extract_at_points(smbu_calc, smb_all)

# --- 4. Linear Regression & Stats ---
# Linear regression: SMB_calc_IGMf ~ SMB
slope, intercept, r_value, p_value, std_err = stats.linregress(smb_all['field_5'], smb_all['SMB_calc_IGMf'])
line_x = np.array([-10, 10])
line_y = slope * line_x + intercept

# Stats (MAE/RMSE/R2)
# Using SMB as reference, SMB_calc_IGMf as prediction
diff = smb_all['field_5'] - smb_all['SMB_calc_IGMf']
mae = np.abs(diff).mean()
rmse = np.sqrt((diff**2).mean())
r_squared = r_value**2

print(f"Stats: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r_squared:.3f}")

# --- 5. Plot Comparison (Stakes) ---
plt.figure(figsize=(8, 8))
plt.errorbar(smb_all['field_5'], smb_all['SMB_calc_IGMf'], yerr=smb_all['SMBu_calc_IGMf'], 
             fmt='ro', label='Stakes all', capsize=3)
plt.plot(line_x, line_y, 'r-', label='Gradient all')
plt.plot([-10, 4], [-10, 4], 'k--', alpha=0.5, label='1:1 Line')

plt.xlim([-9, 4]); plt.ylim([-9, 4])
plt.xlabel('SMB GLACIOCLIM (m a.s.l)')
plt.ylabel('SMB calculated (m w.eq)')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig(os.path.join(results_dir, f"Stakes_IGMf{'_10percbest' if best else ''}.png"))

# --- 6. Altitudinal Plot ---
# This requires masking based on shapefiles
import geopandas as gpd
outlines_whole = gpd.read_file('..\data\gis\outline_pleiades_2020-08-09_utm32N.shp')

# Create mask using rasterio.features.rasterize
mask_whole = features.rasterize(
    [(shape, 1) for shape in outlines_whole.geometry],
    out_shape=smb_calc.shape,
    transform=smb_calc.rio.transform(),
    fill=0
)

plt.figure(figsize=(10, 6))
# Flatten arrays and apply mask
valid_mask = (mask_whole == 1) & (~np.isnan(smb_calc.values))
plt.scatter(dem_r[valid_mask], smb_calc.values[valid_mask], s=1, c='brown', alpha=0.3, label='Calc. SMB')
plt.scatter(smb_all['field_4'], smb_all['field_5'], c='orange', edgecolors='white', label='Stakes')

plt.xlim([1800, 3800]); plt.ylim([-15, 15])
plt.xlabel('Altitude (m a.s.l)')
plt.ylabel('SMB (m w.eq)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, f"Altitudinal_IGMf{'_10percbest' if best else ''}.png"))

plt.show()