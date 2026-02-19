# -*- coding: utf-8 -*-
"""
Preprocess BVVF data (sharkweb long-format export) for GAM trend analysis.

Reads: data_in/sharkweb_data_1990-2024_fyskem_BVVF.txt
Writes: data_in/trenderBVVF_yearly.txt (same format as trenderHAV_update2025MHfix2.txt)

Parameters evaluated:
- NO2, NO3, NH4, DIN (= NO2 + NO3 + NH4)
- PO4, Tot-P, Tot-N, SiO3
- Temperature, Secchi depth, Chlorophyll a, O2

Only yearly trends (helår), surface layer 0-10 m + bottom water O2.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Mapping from long sharkweb column names to short internal names
LONG_TO_SHORT = {
    "station_name": "STATN",
    "sample_location_id": "REG_ID",
    "visit_date": "SDATE",
    "visit_year": "YEAR",
    "visit_month": "MONTH",
    "visit_id": "VISITID",
    "sample_latitude_dd": "LATIT_DD",
    "sample_longitude_dd": "LONGI_DD",
    "sample_depth_m": "DEPH",
    "Temperature bottle (C)": "TEMP_BTL",
    "Temperature CTD (C)": "TEMP_CTD",
    "QFLAG Temperature bottle": "Q_TEMP_BTL",
    "QFLAG Temperature CTD": "Q_TEMP_CTD",
    "Salinity bottle (o/oo psu)": "SALT_BTL",
    "Salinity CTD (o/oo psu)": "SALT_CTD",
    "QFLAG Salinity bottle": "Q_SALT_BTL",
    "QFLAG Salinity CTD": "Q_SALT_CTD",
    "Dissolved oxygen O2 bottle (ml/l)": "DOXY_BTL",
    "Dissolved oxygen O2 CTD (ml/l)": "DOXY_CTD",
    "QFLAG Dissolved oxygen O2 bottle": "Q_DOXY_BTL",
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

# Station name corrections
STATION_RENAMES = {
    "KOSTERFJORDEN": "KOSTERFJORDEN NR16",
}

# REG_ID lookup for stations (from the 1990-2024 sharkweb export)
STATION_REG_ID = {
    "BJÖRKHOLMEN": "156941",
    "BYFJORDEN": "135702",
    "BYTTELOCKET": "156852",
    "DANAFJORD": "156761",
    "E ÄLVSBORGSBRON": "156858",
    "GALTERÖ": "156783",
    "HAVSTENSFJORD": "135701",
    "INSTÖ RÄNNA": "157602",
    "KOLJÖFJORD": "135697",
    "KOSTERFJORDEN NR16": "135542",
    "SKALKORGARNA": "156829",
    "STRETUDDEN": "135279",
    "VALÖ": "156821",
    "ÅSTOL": "135708",
}


def read_file(file_path: Path) -> pd.DataFrame:
    """Read a tab-separated sharkweb/LIMS file (auto-detects encoding)."""
    return pd.read_csv(
        file_path,
        sep="\t",
        encoding="latin1",
        low_memory=False,
    )


def detect_and_rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect header format (long vs short) and rename to short internal names."""
    if "station_name" in df.columns:
        # Long-format sharkweb export
        df = df.rename(columns=LONG_TO_SHORT)
    # else: already short-format (LIMS), columns like STATN, DEPH etc.

    # Correct station names
    for old_name, new_name in STATION_RENAMES.items():
        df.loc[df["STATN"] == old_name, "STATN"] = new_name

    return df


def ensure_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure REG_ID, YEAR, MONTH exist and are properly typed."""
    # Assign REG_ID from lookup if missing
    if "REG_ID" not in df.columns or df["REG_ID"].isna().all():
        df["REG_ID"] = df["STATN"].map(STATION_REG_ID)
    else:
        df["REG_ID"] = pd.to_numeric(df["REG_ID"], errors="coerce").astype("Int64").astype(str)

    # Parse SDATE and derive YEAR/MONTH if needed
    df["SDATE"] = pd.to_datetime(df["SDATE"], errors="coerce")

    if "YEAR" not in df.columns or df["YEAR"].isna().all():
        df["YEAR"] = df["SDATE"].dt.year
    else:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

    if "MONTH" not in df.columns or df["MONTH"].isna().all():
        df["MONTH"] = df["SDATE"].dt.month
    else:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)

    return df


def filter_bad_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Set values to NaN where quality flag is 'B'."""
    qflag_cols = [c for c in df.columns if c.startswith("Q_")]
    for qcol in qflag_cols:
        data_col = qcol[2:]  # remove Q_ prefix
        if data_col in df.columns:
            mask = df[qcol] == "B"
            df.loc[mask, data_col] = np.nan
    return df


def calculate_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived parameters: TEMP, O2, DIN, sumNOx."""
    # TEMP: prefer CTD, fallback to bottle
    df["TEMP"] = df["TEMP_CTD"].where(df["TEMP_CTD"].notna(), df["TEMP_BTL"])

    # O2: prefer bottle, fallback to CTD (exclude low CTD values < 0.2)
    ctd_valid = df["DOXY_CTD"].where(df["DOXY_CTD"] >= 0.2)
    df["O2"] = df["DOXY_BTL"].where(df["DOXY_BTL"].notna(), ctd_valid)

    # sumNOx: prefer NTRZ (NO2+NO3 combined), fallback to NTRA + NTRI
    df["sumNOx"] = df["NTRZ"].where(
        df["NTRZ"].notna(),
        df["NTRA"].add(df["NTRI"], fill_value=0).where(df["NTRA"].notna()),
    )

    # DIN = sumNOx + NH4
    df["DIN"] = df["sumNOx"].add(df["AMON"], fill_value=0).where(
        df["sumNOx"].notna() | df["AMON"].notna()
    )

    return df


def filter_surface(df: pd.DataFrame, max_depth: float = 10.0) -> pd.DataFrame:
    """Filter to surface layer 0-10 m."""
    return df[df["DEPH"].between(0, max_depth)].copy()


def calculate_depth_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean per station, date, and depth interval."""
    value_cols = [
        "TEMP", "O2", "PHOS", "PTOT", "NTRI", "NTRA", "AMON",
        "sumNOx", "DIN", "NTOT", "SIO3-SI", "CPHL", "SECCHI",
    ]
    group_cols = ["STATN", "REG_ID", "YEAR", "MONTH", "SDATE"]

    # Only aggregate columns that exist and have data
    existing_value_cols = [c for c in value_cols if c in df.columns]

    df_depth_mean = (
        df.groupby(group_cols, dropna=False)[existing_value_cols]
        .mean()
        .reset_index()
    )
    return df_depth_mean



# Station-specific bottom water depth thresholds (metres)
BW_THRESHOLDS = {
    "BJÖRKHOLMEN": 50.0,
    "BYFJORDEN": 30.0,
    "BYTTELOCKET": 20.0,
    "DANAFJORD": 30.0,
    "E ÄLVSBORGSBRON": 10.0,
    "GALTERÖ": 30.0,
    "HAVSTENSFJORD": 30.0,
    "INSTÖ RÄNNA": 10.0,
    "KOLJÖFJORD": 30.0,
    "KOSTERFJORDEN NR16": 200.0,
    "SKALKORGARNA": 10.0,
    "STRETUDDEN": 30.0,
    "VALÖ": 20.0,
    "ÅSTOL": 50.0,
}


def extract_bottom_o2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bottom water O2 using station-specific depth thresholds.
    For each station/visit, selects only the deepest O2 measurement
    that is at or below the BW threshold.
    """
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

        # Select only the deepest measurement per visit
        idx_deepest = sub.groupby(["STATN", "REG_ID", "SDATE"])["DEPH"].idxmax()
        deepest = sub.loc[idx_deepest]
        depths = deepest["DEPH"]
        print(f"  {stn}: >= {bw_depth} m -> {len(deepest)} visits, "
              f"deepest depth range {depths.min():.0f}-{depths.max():.0f} m")
        parts.append(deepest)

    if not parts:
        return pd.DataFrame()

    df_bottom = pd.concat(parts)
    return df_bottom[["STATN", "REG_ID", "YEAR", "MONTH", "SDATE", "DEPH", "O2"]]


def calculate_bottom_o2_periods(df_bottom: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate yearly mean bottom water O2 per station for two periods:
      - helår (all months)
      - höst (August-October)
    """
    if df_bottom.empty:
        return pd.DataFrame()

    periods = {
        "helår": None,                   # all months
        "höst": [8, 9, 10],              # August-October
    }

    group_cols = ["STATN", "REG_ID", "YEAR"]
    results = []

    for period_name, months in periods.items():
        if months is not None:
            data = df_bottom[df_bottom["MONTH"].isin(months)]
        else:
            data = df_bottom

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
    """
    Calculate yearly means per station per parameter.
    Returns a long-format DataFrame matching the trenderHAV output format.
    """
    group_cols = ["STATN", "REG_ID", "YEAR"]

    # Parameters to evaluate with their output names
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
        g = data_subset.groupby(group_cols)
        grouped = g.agg(value=(col, "mean"), count=(col, "count")).reset_index()
        # Compute unique months separately (returns array, not scalar)
        unique_months = g["MONTH"].apply(lambda x: np.array(sorted(x.unique())))
        grouped["unique"] = unique_months.values
        grouped = grouped.round(3)

        if grouped.empty:
            continue

        depth_desc = "0-10 m"
        grouped["Depth_interval"] = depth_desc
        grouped["depth_desc"] = depth_desc
        grouped["Mätvariabel"] = f"{var_name} {depth_desc} - helår"
        grouped["variable"] = var_name
        grouped["period"] = "helår"
        results.append(grouped)

    return pd.concat(results, ignore_index=True)


def load_and_normalise(file_path: Path) -> pd.DataFrame:
    """Load a data file, normalise columns, convert types, filter quality."""
    print(f"Reading {file_path}")
    df = read_file(file_path)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    df = detect_and_rename_columns(df)
    print(f"  Stations: {sorted(df['STATN'].unique())}")

    # Convert numeric columns
    numeric_cols = [
        "DEPH", "TEMP_BTL", "TEMP_CTD", "SALT_BTL", "SALT_CTD",
        "DOXY_BTL", "DOXY_CTD", "H2S", "PHOS", "PTOT", "NTRI", "NTRA",
        "NTRZ", "AMON", "NTOT", "SIO3-SI", "CPHL", "SECCHI",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = ensure_common_columns(df)

    print("  Filtering bad quality flags...")
    df = filter_bad_quality(df)

    return df


def main():
    project_root = Path(__file__).resolve().parent.parent
    input_files = [
        project_root / "data_in" / "sharkweb_data_1990-2024_fyskem_BVVF.txt",
        project_root / "data_in" / "data_2025_fyskem_BVVF.txt",
    ]
    output_file = project_root / "data_in" / "trenderBVVF_yearly.txt"

    # Load and merge all input files
    dfs = []
    for f in input_files:
        dfs.append(load_and_normalise(f))
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df)} rows")
    print(f"  Stations: {sorted(df['STATN'].unique())}")
    print(f"  Years: {df['YEAR'].min()}-{df['YEAR'].max()}")

    print("Filtering bad quality flags...")
    df = filter_bad_quality(df)

    print("Calculating derived parameters...")
    df = calculate_parameters(df)

    print("Filtering to surface layer (0-10 m)...")
    df_surface = filter_surface(df)
    print(f"  {len(df_surface)} rows after depth filter")

    print("Calculating depth-interval means per station/date...")
    df_depth_mean = calculate_depth_mean(df_surface)
    print(f"  {len(df_depth_mean)} station-date means")

    print("Extracting bottom water O2 (deepest measurement per visit)...")
    df_bottom = extract_bottom_o2(df)

    print("Calculating yearly means (surface)...")
    result_surface = calculate_yearly_mean(df_depth_mean)

    print("Calculating yearly means (bottom water O2, helår + höst)...")
    result_bottom = calculate_bottom_o2_periods(df_bottom)

    # Combine surface and bottom water results
    result = pd.concat([result_surface, result_bottom], ignore_index=True)

    # Filter: require at least 2 observations per year
    result = result[result["count"] >= 2].reset_index(drop=True)

    # Reorder columns to match expected format
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
