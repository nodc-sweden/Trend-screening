# -*- coding: utf-8 -*-
"""
Preprocess HAL data for GAM trend analysis.

Data sources (in priority order):
1. data_in/HAL/to_use/sharkweb_data_physicalchemical_1993-2025.txt (master, Swedish headers)
2. data_in/HAL/to_use/Anholt_1993-2024.txt (English headers, ANHOLT E only)
3. 2025 cruise folders with Raw_data/data.txt (LIMS short-format, supplementary)

Stations: L9, N5, N6, N7, N13, N14, ANHOLT E  (years 1993-2025)

Output: data_in/trenderHAL_yearly.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Target stations and their REG_IDs
# ---------------------------------------------------------------------------
STATION_REG_ID = {
    "L9 LAHOLMSBUKTEN": "157223",
    "N5 KUNGSBACKAFJORDEN": "157063",
    "N6 KUNGSBACKAFJORDEN": "135584",
    "N7 OST NIDINGEN": "135016",
    "N13 VÄRÖ": "157710",
    "N14 FALKENBERG": "135583",
    "ANHOLT E": "157261",
}

TARGET_STATIONS = set(STATION_REG_ID.keys())

# ---------------------------------------------------------------------------
# Bottom water O2 depth thresholds (metres)
# ---------------------------------------------------------------------------
BW_THRESHOLDS = {
    "N5 KUNGSBACKAFJORDEN": 10.0,
    "N6 KUNGSBACKAFJORDEN": 20.0,
    "N7 OST NIDINGEN": 20.0,
    "N13 VÄRÖ": 15.0,
    "N14 FALKENBERG": 20.0,
    "L9 LAHOLMSBUKTEN": 15.0,
    "ANHOLT E": 40.0,
}

# ---------------------------------------------------------------------------
# Column mapping: Swedish sharkweb headers → short internal names
# ---------------------------------------------------------------------------
SWEDISH_TO_SHORT = {
    "Stationsnamn": "STATN",
    "Nationellt provplats-ID": "REG_ID",
    "Provtagningsdatum": "SDATE",
    "År": "YEAR",
    "Månad": "MONTH",
    "Provtagningsdjup (m)": "DEPH",
    "Temperatur vattenhämtare (C)": "TEMP_BTL",
    "Q-flagga Temperatur vattenhämtare": "Q_TEMP_BTL",
    "Temperatur CTD (C)": "TEMP_CTD",
    "Q-flagga Temperatur CTD": "Q_TEMP_CTD",
    "Salinitet vattenhämtare (o/oo psu)": "SALT_BTL",
    "Q-flagga Salinitet vattenhämtare": "Q_SALT_BTL",
    "Salinitet CTD (o/oo psu)": "SALT_CTD",
    "Q-flagga Salinitet CTD": "Q_SALT_CTD",
    "Syrgashalt O2 vattenhämtare (ml/l)": "DOXY_BTL",
    "Q-flagga Syrgashalt O2 vattenhämtare": "Q_DOXY_BTL",
    "Syrgashalt O2 CTD (ml/l)": "DOXY_CTD",
    "Q-flagga Syrgashalt O2 CTD": "Q_DOXY_CTD",
    "Svavelväte H2S (umol/l)": "H2S",
    "Q-flagga Svavelväte H2S-S": "Q_H2S",
    "Fosfatfosfor PO4-P (umol/l)": "PHOS",
    "Q-flagga Fosfatfosfor PO4-P": "Q_PHOS",
    "Total fosfor Tot-P (umol/l)": "PTOT",
    "Q-flagga Total fosfor Tot-P": "Q_PTOT",
    "Nitritkväve NO2-N (umol/l)": "NTRI",
    "Q-flagga Nitritkväve NO2-N": "Q_NTRI",
    "Nitratkväve NO3-N (umol/l)": "NTRA",
    "Q-flagga Nitratkväve NO3-N": "Q_NTRA",
    "Nitrit+Nitratkväve NO2+NO3-N (umol/l)": "NTRZ",
    "Q-flagga Nitrit+Nitratkväve NO2+NO3-N": "Q_NTRZ",
    "Ammoniumkväve NH4-N (umol/l)": "AMON",
    "Q-flagga Ammoniumkväve NH4-N": "Q_AMON",
    "Total kväve Tot-N (umol/l)": "NTOT",
    "Q-flagga Total kväve Tot-N": "Q_NTOT",
    "Silikat SiO3-Si (umol/l)": "SIO3-SI",
    "Q-flagga Silikat SiO3-Si": "Q_SIO3-SI",
    "Siktdjup (m)": "SECCHI",
    "Q-flagga Siktdjup": "Q_SECCHI",
    "Klorofyll-a vattenhämtare (ug/l)": "CPHL",
    "Q-flagga Klorofyll-a vattenhämtare": "Q_CPHL",
}

# Column mapping: English sharkweb headers → short internal names (Anholt file)
ENGLISH_TO_SHORT = {
    "station_name": "STATN",
    "sample_location_id": "REG_ID",
    "visit_date": "SDATE",
    "visit_year": "YEAR",
    "visit_month": "MONTH",
    "sample_depth_m": "DEPH",
    "Temperature bottle (C)": "TEMP_BTL",
    "QFLAG Temperature bottle": "Q_TEMP_BTL",
    "Temperature CTD (C)": "TEMP_CTD",
    "QFLAG Temperature CTD": "Q_TEMP_CTD",
    "Salinity bottle (o/oo psu)": "SALT_BTL",
    "QFLAG Salinity bottle": "Q_SALT_BTL",
    "Salinity CTD (o/oo psu)": "SALT_CTD",
    "QFLAG Salinity CTD": "Q_SALT_CTD",
    "Dissolved oxygen O2 bottle (ml/l)": "DOXY_BTL",
    "QFLAG Dissolved oxygen O2 bottle": "Q_DOXY_BTL",
    "Dissolved oxygen O2 CTD (ml/l)": "DOXY_CTD",
    "QFLAG Dissolved oxygen O2 CTD": "Q_DOXY_CTD",
    "Hydrogen sulphide H2S (umol/l)": "H2S",
    "QFLAG Hydrogen sulphide H2S": "Q_H2S",
    "Phosphate PO4-P (umol/l)": "PHOS",
    "QFLAG Phosphate PO4-P": "Q_PHOS",
    "Total phosphorus Tot-P (umol/l)": "PTOT",
    "QFLAG Total phosphorus Tot-P": "Q_PTOT",
    "Nitrite NO2-N (umol/l)": "NTRI",
    "QFLAG Nitrite NO2-N": "Q_NTRI",
    "Nitrate NO3-N (umol/l)": "NTRA",
    "QFLAG Nitrate NO3-N": "Q_NTRA",
    "Nitrite+Nitrate NO2+NO3-N (umol/l)": "NTRZ",
    "QFLAG Nitrite+Nitrate NO2+NO3-N": "Q_NTRZ",
    "Ammonium NH4-N (umol/l)": "AMON",
    "QFLAG Ammonium NH4-N": "Q_AMON",
    "Total Nitrogen Tot-N (umol/l)": "NTOT",
    "QFLAG Total Nitrogen Tot-N": "Q_NTOT",
    "Silicate SiO3-Si (umol/l)": "SIO3-SI",
    "QFLAG Silicate SiO3-Si": "Q_SIO3-SI",
    "Secchi depth (m)": "SECCHI",
    "QFLAG Secchi depth": "Q_SECCHI",
    "Chlorophyll-a bottle (ug/l)": "CPHL",
    "QFLAG Chlorophyll-a bottle": "Q_CPHL",
}

# Short internal names used in LIMS/cruise data.txt files (already correct)
LIMS_COLUMNS = [
    "STATN", "SDATE", "MYEAR", "DEPH",
    "TEMP_BTL", "Q_TEMP_BTL", "TEMP_CTD", "Q_TEMP_CTD",
    "SALT_BTL", "Q_SALT_BTL", "SALT_CTD", "Q_SALT_CTD",
    "DOXY_BTL", "Q_DOXY_BTL", "DOXY_CTD", "Q_DOXY_CTD",
    "H2S", "Q_H2S",
    "PHOS", "Q_PHOS", "PTOT", "Q_PTOT",
    "NTRI", "Q_NTRI", "NTRA", "Q_NTRA", "NTRZ", "Q_NTRZ",
    "AMON", "Q_AMON", "NTOT", "Q_NTOT",
    "SIO3-SI", "Q_SIO3-SI",
    "SECCHI", "Q_SECCHI", "CPHL", "Q_CPHL",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_sharkweb_swedish(file_path: Path) -> pd.DataFrame:
    """Read the main sharkweb file (Swedish column headers, cp1252)."""
    print(f"Reading {file_path.name} (Swedish headers)")
    df = pd.read_csv(file_path, sep="\t", encoding="cp1252", low_memory=False)
    print(f"  {len(df)} rows, columns detected: {len(df.columns)}")
    df = df.rename(columns=SWEDISH_TO_SHORT)
    return df


def read_sharkweb_english(file_path: Path) -> pd.DataFrame:
    """Read the Anholt file (English column headers, cp1252)."""
    print(f"Reading {file_path.name} (English headers)")
    df = pd.read_csv(file_path, sep="\t", encoding="cp1252", low_memory=False)
    print(f"  {len(df)} rows")
    df = df.rename(columns=ENGLISH_TO_SHORT)
    return df


def read_lims_cruise(folder: Path) -> pd.DataFrame:
    """Read a LIMS cruise Raw_data/data.txt file (short column names)."""
    data_file = folder / "Raw_data" / "data.txt"
    if not data_file.exists():
        return pd.DataFrame()
    print(f"Reading cruise: {folder.name}")
    df = pd.read_csv(data_file, sep="\t", encoding="cp1252", low_memory=False)
    print(f"  {len(df)} rows, stations: {sorted(df['STATN'].unique())}")

    # Rename MYEAR → YEAR for consistency
    if "MYEAR" in df.columns and "YEAR" not in df.columns:
        df = df.rename(columns={"MYEAR": "YEAR"})

    return df


# ---------------------------------------------------------------------------
# Normalisation & cleaning
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "DEPH", "TEMP_BTL", "TEMP_CTD", "SALT_BTL", "SALT_CTD",
    "DOXY_BTL", "DOXY_CTD", "H2S", "PHOS", "PTOT", "NTRI", "NTRA",
    "NTRZ", "AMON", "NTOT", "SIO3-SI", "CPHL", "SECCHI",
]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, convert types, filter to target stations."""
    # Convert numeric columns (handles '<0.20' style values → NaN)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse date
    df["SDATE"] = pd.to_datetime(df["SDATE"], errors="coerce")

    if "YEAR" not in df.columns or df["YEAR"].isna().all():
        df["YEAR"] = df["SDATE"].dt.year
    else:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

    if "MONTH" not in df.columns or df["MONTH"].isna().all():
        df["MONTH"] = df["SDATE"].dt.month
    else:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    # Assign REG_ID from lookup
    df["REG_ID"] = df["STATN"].map(STATION_REG_ID)

    # Filter to target stations and year range
    df = df[df["STATN"].isin(TARGET_STATIONS)].copy()
    df = df[(df["YEAR"] >= 1993) & (df["YEAR"] <= 2025)].copy()

    if df.empty:
        return df

    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)

    return df


def filter_bad_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Set values to NaN where quality flag is 'B'."""
    qflag_cols = [c for c in df.columns if c.startswith("Q_")]
    for qcol in qflag_cols:
        data_col = qcol[2:]
        if data_col in df.columns:
            mask = df[qcol] == "B"
            df.loc[mask, data_col] = np.nan
    return df


def calculate_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived parameters: TEMP, O2, DIN, sumNOx."""
    # TEMP: prefer CTD, fallback to bottle
    if "TEMP_CTD" in df.columns and "TEMP_BTL" in df.columns:
        df["TEMP"] = df["TEMP_CTD"].where(df["TEMP_CTD"].notna(), df["TEMP_BTL"])
    elif "TEMP_CTD" in df.columns:
        df["TEMP"] = df["TEMP_CTD"]
    elif "TEMP_BTL" in df.columns:
        df["TEMP"] = df["TEMP_BTL"]

    # O2: prefer bottle, fallback to CTD (exclude low CTD values < 0.2)
    if "DOXY_CTD" in df.columns:
        ctd_valid = df["DOXY_CTD"].where(df["DOXY_CTD"] >= 0.2)
    else:
        ctd_valid = pd.Series(np.nan, index=df.index)

    if "DOXY_BTL" in df.columns:
        df["O2"] = df["DOXY_BTL"].where(df["DOXY_BTL"].notna(), ctd_valid)
    else:
        df["O2"] = ctd_valid

    # sumNOx: prefer NTRZ (NO2+NO3 combined), fallback to NTRA + NTRI
    if "NTRZ" in df.columns:
        df["sumNOx"] = df["NTRZ"].copy()
    else:
        df["sumNOx"] = pd.Series(np.nan, index=df.index)

    if "NTRA" in df.columns and "NTRI" in df.columns:
        fallback = df["NTRA"].add(df["NTRI"], fill_value=0).where(df["NTRA"].notna())
        df["sumNOx"] = df["sumNOx"].where(df["sumNOx"].notna(), fallback)

    # DIN = sumNOx + NH4
    if "AMON" in df.columns:
        df["DIN"] = df["sumNOx"].add(df["AMON"], fill_value=0).where(
            df["sumNOx"].notna() | df["AMON"].notna()
        )
    else:
        df["DIN"] = df["sumNOx"]

    return df


# ---------------------------------------------------------------------------
# Aggregation (identical logic to preprocess_BVVF.py)
# ---------------------------------------------------------------------------

def filter_surface(df: pd.DataFrame, max_depth: float = 10.0) -> pd.DataFrame:
    return df[df["DEPH"].between(0, max_depth)].copy()


def calculate_depth_mean(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "TEMP", "O2", "PHOS", "PTOT", "NTRI", "NTRA", "AMON",
        "sumNOx", "DIN", "NTOT", "SIO3-SI", "CPHL", "SECCHI",
    ]
    group_cols = ["STATN", "REG_ID", "YEAR", "MONTH", "SDATE"]
    existing = [c for c in value_cols if c in df.columns]
    return (
        df.groupby(group_cols, dropna=False)[existing]
        .mean()
        .reset_index()
    )


def extract_bottom_o2(df: pd.DataFrame) -> pd.DataFrame:
    df_o2 = df.dropna(subset=["O2", "DEPH"]).copy()
    if df_o2.empty:
        return pd.DataFrame()

    parts = []
    for stn, bw_depth in BW_THRESHOLDS.items():
        mask = (df_o2["STATN"] == stn) & (df_o2["DEPH"] >= bw_depth)
        sub = df_o2.loc[mask]
        if sub.empty:
            print(f"  {stn}: no O2 data >= {bw_depth} m")
            continue

        idx_deepest = sub.groupby(["STATN", "REG_ID", "SDATE"])["DEPH"].idxmax()
        deepest = sub.loc[idx_deepest]
        depths = deepest["DEPH"]
        print(f"  {stn}: >= {bw_depth} m -> {len(deepest)} visits, "
              f"depth range {depths.min():.0f}-{depths.max():.0f} m")
        parts.append(deepest)

    if not parts:
        return pd.DataFrame()

    df_bottom = pd.concat(parts)
    return df_bottom[["STATN", "REG_ID", "YEAR", "MONTH", "SDATE", "DEPH", "O2"]]


def calculate_bottom_o2_periods(df_bottom: pd.DataFrame) -> pd.DataFrame:
    if df_bottom.empty:
        return pd.DataFrame()

    periods = {
        "helår": None,
        "höst": [8, 9, 10],
    }
    group_cols = ["STATN", "REG_ID", "YEAR"]
    results = []

    for period_name, months in periods.items():
        data = df_bottom if months is None else df_bottom[df_bottom["MONTH"].isin(months)]
        if data.empty:
            continue

        g = data.groupby(group_cols)
        grouped = g.agg(value=("O2", "mean"), count=("O2", "count")).reset_index()
        unique_months = g["MONTH"].apply(lambda x: np.array(sorted(x.unique())))
        grouped["unique"] = unique_months.values
        grouped = grouped.round(3)

        grouped["Depth_interval"] = "Bottenvatten"
        grouped["depth_desc"] = "Bottenvatten"
        grouped["Mätvariabel"] = f"O2 Bottenvatten - {period_name}"
        grouped["variable"] = "O2"
        grouped["period"] = period_name
        results.append(grouped)

    return pd.concat(results, ignore_index=True)


def calculate_yearly_mean(df_depth_mean: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["STATN", "REG_ID", "YEAR"]

    parameters = {
        "NTRI": "NO2",
        "NTRA": "NO3",
        "AMON": "NH4",
        "DIN": "DIN",
        "NTOT": "TOTN",
        "PHOS": "PO4",
        "PTOT": "TOTP",
        "SIO3-SI": "SiO3",
        "SECCHI": "SECCHI",
        "TEMP": "TEMPERATUR",
        "O2": "O2",
        "CPHL": "CHL a",
    }

    results = []
    for col, var_name in parameters.items():
        if col not in df_depth_mean.columns:
            continue

        data_subset = df_depth_mean.dropna(subset=[col])
        if data_subset.empty:
            continue

        g = data_subset.groupby(group_cols)
        grouped = g.agg(value=(col, "mean"), count=(col, "count")).reset_index()
        unique_months = g["MONTH"].apply(lambda x: np.array(sorted(x.unique())))
        grouped["unique"] = unique_months.values
        grouped = grouped.round(3)

        depth_desc = "0-10 m"
        grouped["Depth_interval"] = depth_desc
        grouped["depth_desc"] = depth_desc
        grouped["Mätvariabel"] = f"{var_name} {depth_desc} - helår"
        grouped["variable"] = var_name
        grouped["period"] = "helår"
        results.append(grouped)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(master: pd.DataFrame, supplement: pd.DataFrame) -> pd.DataFrame:
    """
    Add rows from supplement that don't exist in master.
    Match on STATN + SDATE + DEPH to identify duplicates.
    """
    if supplement.empty:
        return master
    if master.empty:
        return supplement

    master_keys = set(
        zip(master["STATN"], master["SDATE"].astype(str), master["DEPH"])
    )
    mask = [
        (s, str(d), dep) not in master_keys
        for s, d, dep in zip(supplement["STATN"], supplement["SDATE"], supplement["DEPH"])
    ]
    new_rows = supplement[mask]
    print(f"  Dedup: {len(supplement)} supplement rows -> {new_rows.shape[0]} new rows added")
    return pd.concat([master, new_rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    hal_dir = project_root / "data_in" / "HAL" / "to_use"
    output_file = project_root / "data_in" / "trenderHAL_yearly.txt"

    # --- 1. Load master sharkweb file (Swedish headers) ---
    df_master = read_sharkweb_swedish(
        hal_dir / "sharkweb_data_physicalchemical_1993-2025.txt"
    )
    df_master = normalise(df_master)
    print(f"  Master: {len(df_master)} rows, stations: {sorted(df_master['STATN'].unique())}")

    # --- 2. Load Anholt file (English headers) ---
    df_anholt = read_sharkweb_english(hal_dir / "Anholt_1993-2024.txt")
    df_anholt = normalise(df_anholt)
    print(f"  Anholt: {len(df_anholt)} rows")

    # Merge Anholt into master (deduplicate)
    print("Merging Anholt into master...")
    df_master = deduplicate(df_master, df_anholt)

    # --- 3. Load 2025 cruise data (supplementary) ---
    cruise_folders = sorted(hal_dir.glob("2026-*"))
    for folder in cruise_folders:
        df_cruise = read_lims_cruise(folder)
        if df_cruise.empty:
            continue
        df_cruise = normalise(df_cruise)
        if df_cruise.empty:
            continue
        print(f"  Merging cruise data from {folder.name}...")
        df_master = deduplicate(df_master, df_cruise)

    df = df_master
    print(f"\nCombined: {len(df)} rows")
    print(f"  Stations: {sorted(df['STATN'].unique())}")
    print(f"  Years: {df['YEAR'].min()}-{df['YEAR'].max()}")

    # --- 4. Quality filtering ---
    print("Filtering bad quality flags...")
    df = filter_bad_quality(df)

    # --- 5. Derived parameters ---
    print("Calculating derived parameters...")
    df = calculate_parameters(df)

    # --- 6. Surface layer ---
    print("Filtering to surface layer (0-10 m)...")
    df_surface = filter_surface(df)
    print(f"  {len(df_surface)} rows after depth filter")

    print("Calculating depth-interval means per station/date...")
    df_depth_mean = calculate_depth_mean(df_surface)
    print(f"  {len(df_depth_mean)} station-date means")

    # --- 7. Bottom water O2 ---
    print("Extracting bottom water O2...")
    df_bottom = extract_bottom_o2(df)

    # --- 8. Yearly means ---
    print("Calculating yearly means (surface)...")
    result_surface = calculate_yearly_mean(df_depth_mean)

    print("Calculating yearly means (bottom water O2, helår + höst)...")
    result_bottom = calculate_bottom_o2_periods(df_bottom)

    # --- 9. Combine and output ---
    result = pd.concat([result_surface, result_bottom], ignore_index=True)

    # Require at least 2 observations per year
    result = result[result["count"] >= 2].reset_index(drop=True)

    output_cols = [
        "STATN", "REG_ID", "YEAR", "Depth_interval",
        "value", "count", "unique", "depth_desc",
        "Mätvariabel", "variable", "period",
    ]
    result = result[output_cols]

    print(f"\nWriting {len(result)} rows to {output_file}")
    print(f"  Stations: {sorted(result['STATN'].unique())}")
    print(f"  Variables: {sorted(result['variable'].unique())}")
    print(f"  Year range: {result['YEAR'].min()}-{result['YEAR'].max()}")

    result.to_csv(
        output_file,
        sep="\t",
        encoding="utf-8",
        index=False,
        float_format="%.2f",
    )
    print("Done!")


if __name__ == "__main__":
    main()
