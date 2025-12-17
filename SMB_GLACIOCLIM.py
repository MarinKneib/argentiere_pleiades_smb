# Initialize
import os
import numpy as np
import pandas as pd
import pyproj
from pyproj import Transformer



# working directory (equivalent to cd)
os.chdir(r"C:\Users\kneibm\Documents\Projects\PI\2024_CAIRN-GLOBAL\SMB_inversions\Bossons\code")

# folders
datafolder = "../data/velocity/PLEIADES_ALPS/"
resultsfolder = "../output/smb_glacioclim/"

os.makedirs(resultsfolder, exist_ok=True)

# CSV loading options (MATLAB opts equivalent)
usecols = [
    "profile_name",
    "stake_number",
    "x_lambert2e",
    "y_lambert2e",
    "altitude",
    "annual_smb",
]

# Initialize arrays
n_stakes = 50
n_years = 10  # 2012–2021

SMB_abl = np.zeros((n_stakes, 7))
SMB_acc = np.zeros((n_stakes, 7))
SMB_les = np.zeros((n_stakes, 7))
SMB_tac = np.zeros((n_stakes, 7))

All_abl = np.full((n_stakes, n_years), np.nan)
All_acc = np.full((n_stakes, n_years), np.nan)
All_les = np.full((n_stakes, n_years), np.nan)
All_tac = np.full((n_stakes, n_years), np.nan)

N_abl = np.zeros(n_stakes, dtype=int)
N_acc = np.zeros(n_stakes, dtype=int)
N_les = np.zeros(n_stakes, dtype=int)
N_tac = np.zeros(n_stakes, dtype=int)

# Loop over years
for iy, yr in enumerate(range(2012, 2022)):

    SMB_abl_yr = pd.read_csv(
        f"../data/smb_glacioclim/Langue/mdg_langue_annual_smb_abl_{yr}.csv",
        usecols=usecols
    )
    SMB_acc_yr = pd.read_csv(
        f"../data/smb_glacioclim/Accumulation/mdg_annual_accu_{yr}.csv",
        usecols=usecols
    )
    SMB_les_yr = pd.read_csv(
        f"../data/smb_glacioclim/Leschaux/mdg_Leschaux_annual_smb_abl_{yr}.csv",
        usecols=usecols
    )
    SMB_tac_yr = pd.read_csv(
        f"../data/smb_glacioclim/Tacul/mdg_Tacul_annual_smb_abl_{yr}.csv",
        usecols=usecols
    )

    for ss in range(1, n_stakes + 1):
        idx = ss - 1

        # --- ABLATION ---
        meas = SMB_abl_yr["stake_number"] == ss
        if meas.any():
            N_abl[idx] += 1
            SMB_abl[idx, 1] = ss
            SMB_abl[idx, 2] += SMB_abl_yr.loc[meas, "x_lambert2e"].mean()
            SMB_abl[idx, 3] += SMB_abl_yr.loc[meas, "y_lambert2e"].mean()
            SMB_abl[idx, 4] += SMB_abl_yr.loc[meas, "altitude"].mean()
            SMB_abl[idx, 5] += SMB_abl_yr.loc[meas, "annual_smb"].mean()
            All_abl[idx, iy] = SMB_abl_yr.loc[meas, "annual_smb"].mean()

        # --- ACCUMULATION ---
        meas = SMB_acc_yr["stake_number"] == ss
        if meas.any():
            N_acc[idx] += 1
            SMB_acc[idx, 1] = ss
            SMB_acc[idx, 2] += SMB_acc_yr.loc[meas, "x_lambert2e"].mean()
            SMB_acc[idx, 3] += SMB_acc_yr.loc[meas, "y_lambert2e"].mean()
            SMB_acc[idx, 4] += SMB_acc_yr.loc[meas, "altitude"].mean()
            SMB_acc[idx, 5] += SMB_acc_yr.loc[meas, "annual_smb"].mean()
            All_acc[idx, iy] = SMB_acc_yr.loc[meas, "annual_smb"].mean()

        # --- LESCHAUX ---
        meas = SMB_les_yr["stake_number"] == ss
        if meas.any():
            N_les[idx] += 1
            SMB_les[idx, 1] = ss
            SMB_les[idx, 2] += SMB_les_yr.loc[meas, "x_lambert2e"].mean()
            SMB_les[idx, 3] += SMB_les_yr.loc[meas, "y_lambert2e"].mean()
            SMB_les[idx, 4] += SMB_les_yr.loc[meas, "altitude"].mean()
            SMB_les[idx, 5] += SMB_les_yr.loc[meas, "annual_smb"].mean()
            All_les[idx, iy] = SMB_les_yr.loc[meas, "annual_smb"].mean()

        # --- TACUL ---
        meas = SMB_tac_yr["stake_number"] == ss
        if meas.any():
            N_tac[idx] += 1
            SMB_tac[idx, 1] = ss
            SMB_tac[idx, 2] += SMB_tac_yr.loc[meas, "x_lambert2e"].mean()
            SMB_tac[idx, 3] += SMB_tac_yr.loc[meas, "y_lambert2e"].mean()
            SMB_tac[idx, 4] += SMB_tac_yr.loc[meas, "altitude"].mean()
            SMB_tac[idx, 5] += SMB_tac_yr.loc[meas, "annual_smb"].mean()
            All_tac[idx, iy] = SMB_tac_yr.loc[meas, "annual_smb"].mean()

# Remove stakes with < 50% measurements and compute averages + std
def finalize(SMB, N, All):
    for i in range(len(N)):
        if N[i] < 6:
            SMB[i, :] = np.nan
        else:
            SMB[i, 2:6] /= N[i]
            SMB[i, 6] = np.nanstd(All[i, :])
    return SMB

SMB_abl = finalize(SMB_abl, N_abl, All_abl)
SMB_acc = finalize(SMB_acc, N_acc, All_acc)
SMB_les = finalize(SMB_les, N_les, All_les)
SMB_tac = finalize(SMB_tac, N_tac, All_tac)

# Lambert II étendu → UTM 32N
transformer_27562_to_32632 = Transformer.from_crs(
    27562, 32632, always_xy=True
)

# Remove NaNs + add UTM coordinates + column names
def clean_and_format(SMB):
    # Drop first column (dummy / unused)
    SMB = SMB[:, 1:]

    # Remove rows with NaNs
    SMB = SMB[~np.isnan(SMB).any(axis=1)]

    # Convert to DataFrame with column names
    df = pd.DataFrame(
        SMB,
        columns=[
            "stake_number",
            "x_lambert2e",
            "y_lambert2e",
            "altitude",
            "mean_annual_smb",
            "std_annual_smb",
        ],
    )

    # Convert coordinates to UTM 32N
    x_utm, y_utm = transformer_27562_to_32632.transform(
        df["x_lambert2e"].values,
        df["y_lambert2e"].values,
    )

    df["x_utm32n"] = x_utm
    df["y_utm32n"] = y_utm

    # Reorder columns
    df = df[
        [
            "stake_number",
            "x_lambert2e",
            "y_lambert2e",
            "x_utm32n",
            "y_utm32n",
            "altitude",
            "mean_annual_smb",
            "std_annual_smb",
        ]
    ]

    return df

SMB_abl = clean_and_format(SMB_abl)
SMB_acc = clean_and_format(SMB_acc)
SMB_les = clean_and_format(SMB_les)
SMB_tac = clean_and_format(SMB_tac)

SMB_abl.to_csv(
    os.path.join(resultsfolder, "SMB_ABL_2012-2021_epsg27562_epsg32632.csv"),
    index=False,
)
SMB_acc.to_csv(
    os.path.join(resultsfolder, "SMB_ACC_2012-2021_epsg27562_epsg32632.csv"),
    index=False,
)
SMB_les.to_csv(
    os.path.join(resultsfolder, "SMB_LES_2012-2021_epsg27562_epsg32632.csv"),
    index=False,
)
SMB_tac.to_csv(
    os.path.join(resultsfolder, "SMB_TAC_2012-2021_epsg27562_epsg32632.csv"),
    index=False,
)

