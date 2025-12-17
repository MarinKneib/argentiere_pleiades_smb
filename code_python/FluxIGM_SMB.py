import os
import glob
import random
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from datetime import datetime
from scipy.interpolate import griddata
from rasterio.enums import Resampling
from rasterio import features
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import csv

# --- Useful stuff ---
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

def save_geotiff(da, filename):
    da.rio.to_raster(os.path.join(outdir, filename), compress='deflate')

# --- Settings & Paths ---
homedir = r'C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\code'
os.chdir(homedir)

# Input Paths (to be changed as needed)
DH_path = '../output/dh_results/tendancy_filt.tif'
DEM_path = '../output/dh_results/meanDEM-2017_02_15.tif'
FDIV_parent_dir = '../igm/run_ite_2025_12_17/'
outline_shp = '../data/gis/BossonsTaconnaz_rgi6_utm32n.shp'

# Parameters - potentially to change!!!
class Params:
    Glacier = 'Bossons'
    DX = 20
    uncertainty = 100  # N runs
    density_mixedzone = 0.75
    sigdensity_mixedzone = 0.15
    sigdH = 0.07
    Qdensity = 0.9

# Set to True if you want to save every single map
export_indiv = True

P = Params()

# Output setup
date_str = datetime.now().strftime("%Y-%m-%d")
outdir = f"../output/smb/{date_str}_fluxIGM_uncertainty_ite_N{P.uncertainty}"
os.makedirs(outdir, exist_ok=True)

# Prepare the header for our log file
log_data = []
log_filename = os.path.join(outdir, 'run_parameters_log.csv')

# Check for 'output_n' subfolders in the FDIV parent dir - in which case the MC loop will randomly draw fdiv from these folders at each iteration
subfolders = glob.glob(os.path.join(FDIV_parent_dir, 'output_*'))

fdiv_paths = []

if subfolders:
    # Scenario A: Multiple folders (output_0, output_1, etc.)
    # We create a list of the specific NetCDF file inside each folder
    for f in subfolders:
        nc_file = os.path.join(f, 'geology-optimized.nc')
        # Only add to the list if the file actually exists
        if os.path.exists(nc_file):
            fdiv_paths.append(nc_file)
        else:
            print(f"Warning: No NetCDF found in {f}, skipping.")
else:
    # Scenario B: Single NetCDF in the root
    root_nc = os.path.join(FDIV_parent_dir, 'geology-optimized.nc')
    if os.path.exists(root_nc):
        fdiv_paths = [root_nc]

# Safety check: make sure we found at least one valid file
if not fdiv_paths:
    raise FileNotFoundError("No valid 'geology-optimized.nc' files were found in the specified paths.")

print(f"Total valid flux divergence files available: {len(fdiv_paths)}")

# --- Load Data ---

# --- MASTER REFERENCE STEP ---
# Load the first fdiv just to get the bounds
with xr.open_dataset(fdiv_paths[0]) as ds_ref:
    fdiv_ref = ds_ref['divflux'].rio.write_crs("EPSG:32632")
    # Get bounds for clipping other files
    fdiv_bounds = fdiv_ref.rio.bounds()

# Load DEM and dH
dem_ds = rioxarray.open_rasterio(DEM_path).squeeze()
dh_ds = rioxarray.open_rasterio(DH_path).squeeze()
outlines = gpd.read_file(outline_shp)

# --- Alignment & Resampling ---

# Define the target grid (Resolution P.DX)
# We use the FDIV extent as the master domain
minx, miny, maxx, maxy = fdiv_ref.rio.bounds()
x_coords = np.arange(minx, maxx + P.DX, P.DX)
y_coords = np.arange(maxy, miny - P.DX, -P.DX) # North to South

target_grid = xr.DataArray(
    np.zeros((len(y_coords), len(x_coords))),
    coords={'y': y_coords, 'x': x_coords},
    dims=('y', 'x')
).rio.write_crs(fdiv_ref.rio.crs)

# Resample DEM & DH to target grid (average as we're downsampling)
N_DEM = dem_ds.rio.reproject_match(target_grid, resampling=Resampling.average)
N_DH = dh_ds.rio.reproject_match(target_grid, resampling=Resampling.average)

plt.figure(figsize=(6, 5))
im = plt.imshow(N_DH, cmap=cmap_tend, vmin=-6, vmax=6)
plt.colorbar(im, label="elevation chage (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

# Create Mask from outlines
glacier_mask = features.rasterize(
    [(shape, 1) for shape in outlines.geometry],
    out_shape=target_grid.shape,
    transform=target_grid.rio.transform(),
    fill=0
)
N_MASK = xr.DataArray(glacier_mask, coords=target_grid.coords, dims=target_grid.dims)

plt.imshow(N_MASK)

# --- SMB Calculation (Monte Carlo) ---

if P.uncertainty > 0:
    # Initialize stacks
    smb_stack = []
    fdiv_stack = []
    h_dens_stack = []

    for i in range(P.uncertainty):
        # randomly select a flux divergence path
        selected_path = random.choice(fdiv_paths)
        folder_path = os.path.dirname(selected_path)
        json_path = os.path.join(folder_path, 'params.json')

        # Open and process the specific NetCDF for this run
        ds_iter = xr.open_dataset(selected_path)
        fdiv_iter = ds_iter['divflux'].rio.write_crs("EPSG:32632")

        # Resample this specific fdiv to master target grid
        N_FDIV_iter = fdiv_iter.rio.reproject_match(target_grid, resampling=Resampling.average)
        
        # Extract specific JSON params
        json_params = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                full_json = json.load(f)
                # Only record the ones you specified
                keys_to_extract = ["opti_usurfobs_std", "opti_velsurfobs_std", "opti_thkobs_std"]
                json_params = {k: full_json.get(k, "N/A") for k in keys_to_extract}

        # Clean FDIV offset
        fdiv_valid = N_FDIV_iter.where(N_MASK == 1)
        offset = fdiv_valid.mean(skipna=True).item()
        N_FDIV_iter = N_FDIV_iter - offset
        print(f'corrected offset of {offset:.3f} m/yr')

        # Perturbations
        dh_add = np.random.normal(0, P.sigdH)
        dens_mixed = P.density_mixedzone + np.random.uniform(-P.sigdensity_mixedzone, P.sigdensity_mixedzone)
        
        curr_dh = N_DH + dh_add
        
        # Density Logic (Vectorized)
        # 0.9 = melt/ice, 0.6 = firn/acc
        h_density = xr.full_like(N_DEM, 0.9)
        
        ind1 = N_FDIV_iter > 0
        ind2 = curr_dh > 0
        ind3 = np.abs(curr_dh) > np.abs(N_FDIV_iter)
        ind4 = N_DEM > N_DEM.where(N_MASK == 1).median()

        # Apply specific gravity rules
        h_density = xr.where(ind1 & ~ind2, 0.9, h_density)
        h_density = xr.where(~ind1 & ind2, 0.6, h_density)
        h_density = xr.where(ind1 & ind2 & ind3, 0.6, h_density)
        h_density = xr.where(ind1 & ind2 & ~ind3, dens_mixed, h_density)
        h_density = xr.where(~ind1 & ~ind2 & ind3, 0.9, h_density)
        h_density = xr.where(~ind1 & ~ind2 & ~ind3, dens_mixed, h_density)
        h_density = xr.where((~ind4) & (N_MASK == 1), 0.9, h_density)
        
        # Continuity Equation
        smb = (h_density * curr_dh) + (P.Qdensity * N_FDIV_iter)
        
        # Masking
        smb = smb.where(N_MASK == 1)
        
        smb_stack.append(smb)
        h_dens_stack.append(h_density)
        fdiv_stack.append(N_FDIV_iter)

        # Individual exports (if required)
        if export_indiv:
            run_id = f"run_{i:03d}"
            # Save individual GeoTIFFs
            save_geotiff(smb, f"{run_id}_SMB.tif")
            save_geotiff(N_FDIV_iter, f"{run_id}_FDIV.tif")
            save_geotiff(h_density, f"{run_id}_Density.tif")
            
            # Record parameters for this specific run
            run_record = {
                "run_id": run_id,
                "source_folder": os.path.basename(folder_path),
                "sensitivity_dh_add": dh_add,
                "sensitivity_dens_mixed": dens_mixed,
                **json_params # Unpacks the 3 JSON parameters here
            }
            log_data.append(run_record)

        # Keep track for the master mean/std calculation
        smb_stack.append(smb)

        # Save the Log File 
        if export_indiv:
            keys = log_data[0].keys()
            with open(log_filename, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(log_data)
            print(f"Parameter log saved to {log_filename}")


    # Compute Statistics
    SMB_mean = xr.concat(smb_stack, dim='run').mean('run')
    SMB_std = xr.concat(smb_stack, dim='run').std('run')
    FDIV_mean = xr.concat(fdiv_stack, dim='run').mean('run')
    FDIV_std = xr.concat(fdiv_stack, dim='run').std('run')
else:
    # Simple single run logic here (omitted for brevity)
    pass

# --- Plot ---
plt.figure(figsize=(6, 5))
im = plt.imshow(SMB_mean, cmap=cmap_tend, vmin=-6, vmax=6)
plt.colorbar(im, label="SMB (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

plt.figure(figsize=(6, 5))
im = plt.imshow(SMB_std, cmap=cmap_slope, vmin=0, vmax=6)
plt.colorbar(im, label="SMB uncertainty (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

plt.figure(figsize=(6, 5))
im = plt.imshow(FDIV_mean, cmap=cmap_tend, vmin=-6, vmax=6)
plt.colorbar(im, label="FDIV (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

plt.figure(figsize=(6, 5))
im = plt.imshow(FDIV_std, cmap=cmap_slope, vmin=0, vmax=6)
plt.colorbar(im, label="FDIV uncertainty (m yr⁻¹)")
plt.axis("off")
plt.tight_layout()

# --- Export ---


save_geotiff(SMB_mean, f"{P.Glacier}_SMB.tif")
save_geotiff(SMB_std, f"{P.Glacier}_SMBu.tif")
save_geotiff(FDIV_mean, f"{P.Glacier}_FDIV.tif")
save_geotiff(FDIV_std, f"{P.Glacier}_FDIVu.tif")

print(f"Finished. Outputs saved to {outdir}")

