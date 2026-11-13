"""
build_unified.py  (v4 — ZCTA-anchored + tract-level ACS/SVI equity layer)

Outputs:
    data/unified/hospital_risk_matrix.parquet   primary, ZCTA-keyed
                                                 + state-level SVI averages
    data/unified/tract_equity.parquet           secondary, tract-keyed
                                                 + ACS demographics + SVI scores
    data/unified/county_equity.parquet          state/county rollups for the UI

Architecture decision (don't re-pivot): ZCTA primary, tract secondary.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_ROOT = Path("/lstr/sahara/datalab-ml/ibrahim/limagents_update/med_limagnets/data")
OUT_DIR = DATA_ROOT / "unified"
OUT_DIR.mkdir(exist_ok=True)

PLACES_MEASURES = {
    "DIABETES": "diabetes_prev", "CHD": "chd_prev", "COPD": "copd_prev",
    "OBESITY": "obesity_prev", "STROKE": "stroke_prev", "BPHIGH": "bp_high_prev",
    "KIDNEY": "kidney_disease_prev", "CASTHMA": "asthma_prev", "CANCER": "cancer_prev",
    "DEPRESSION": "depression_prev", "ACCESS2": "no_insurance_prev",
    "CHECKUP": "routine_checkup_prev", "MAMMOUSE": "mammogram_prev",
    "COLON_SCREEN": "colon_screen_prev",
}

# ACS variables we extract from each state-tract JSON
ACS_VARS = {
    "B01003_001E": "total_population",
    "B02001_002E": "pop_white",
    "B02001_003E": "pop_black",
    "B02001_005E": "pop_asian",
    "B03003_003E": "pop_hispanic",
    "B01001_020E": "pop_male_65_plus",
    "B01001_044E": "pop_female_65_plus",
}

# Census state FIPS → 2-letter abbreviation (for joining to CMS)
STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
    "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
    "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
    "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
    "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY",
}


def _safe_numeric(s): return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# CDC PLACES
# ---------------------------------------------------------------------------
def _pivot_places(df, geo_col):
    df = df[df["MeasureId"].isin(PLACES_MEASURES.keys())].copy()
    df = df[df["Data_Value_Type"].str.contains("Crude", na=False)]
    return df.pivot_table(index=geo_col, columns="MeasureId",
                          values="Data_Value", aggfunc="mean").reset_index(
                          ).rename(columns=PLACES_MEASURES)


def load_places_zcta():
    df = pd.read_csv(DATA_ROOT / "cdc_places" / "places_zcta.csv", low_memory=False)
    zcta_col = next(c for c in ["ZCTA5","LocationName","LocationID"] if c in df.columns)
    out = _pivot_places(df, zcta_col).rename(columns={zcta_col: "zcta5"})
    out["zcta5"] = out["zcta5"].astype(str).str.zfill(5)
    return out


def load_places_tract():
    df = pd.read_csv(DATA_ROOT / "cdc_places" / "places_tract.csv", low_memory=False)
    tract_col = next(c for c in ["TractFIPS","LocationID"] if c in df.columns)
    out = _pivot_places(df, tract_col).rename(columns={tract_col: "tract_fips"})
    out["tract_fips"] = out["tract_fips"].astype(str).str.zfill(11)
    out["county_fips"] = out["tract_fips"].str[:5]
    out["state_fips"] = out["tract_fips"].str[:2]
    return out


# ---------------------------------------------------------------------------
# CMS Hospital Compare (4 files)
# ---------------------------------------------------------------------------
def load_cms_hospitals():
    base = DATA_ROOT / "cms_hospital_compare"
    gen = pd.read_csv(base / "Hospital_General_Information.csv",
                      low_memory=False, dtype=str)
    keep = [c for c in ["Facility ID","Facility Name","City","State","ZIP Code",
                        "Hospital Type","Hospital Ownership","Emergency Services"]
            if c in gen.columns]
    gen = gen[keep].rename(columns={"Facility ID":"facility_id","ZIP Code":"zcta5"})
    gen["zcta5"] = gen["zcta5"].astype(str).str.zfill(5)

    def _agg(fname, out_col):
        df = pd.read_csv(base / fname, low_memory=False
                         ).rename(columns={"Facility ID":"facility_id"})
        df["Score"] = _safe_numeric(df["Score"])
        return df.groupby("facility_id")["Score"].mean().reset_index(
            ).rename(columns={"Score": out_col})

    out = gen.merge(_agg("Complications_and_Deaths_-_Hospital.csv",
                         "facility_complication_index"), on="facility_id", how="left")
    out = out.merge(_agg("Healthcare_Associated_Infections_-_Hospital.csv",
                         "facility_hai_index"), on="facility_id", how="left")
    out = out.merge(_agg("Unplanned_Hospital_Visits_-_Hospital.csv",
                         "facility_readmission_index"), on="facility_id", how="left")
    return out


# ---------------------------------------------------------------------------
# HRSA — primary care + mental health
# ---------------------------------------------------------------------------
def _hrsa_one(filename, label):
    df = pd.read_csv(DATA_ROOT / "hrsa_shortage_areas" / filename, low_memory=False)
    status_col = next((c for c in df.columns if "Status" in c), None)
    if status_col:
        df = df[df[status_col].astype(str).str.lower().str.contains("designated", na=False)]
    zip_col = next((c for c in df.columns if "ZIP" in c.upper() or "POSTAL" in c.upper()), None)
    score_col = next((c for c in df.columns if "Score" in c and "HPSA" in c), None)
    if zip_col is None:
        return pd.DataFrame(columns=["zcta5", f"hpsa_{label}_flag"])
    df[zip_col] = df[zip_col].astype(str).str.split(".").str[0].str.zfill(5)
    agg = df.groupby(zip_col).size().reset_index(name="n"
        ).rename(columns={zip_col:"zcta5"}).drop(columns="n")
    agg[f"hpsa_{label}_flag"] = 1
    if score_col is not None:
        sc = df.groupby(zip_col)[score_col].max().reset_index(
            ).rename(columns={zip_col:"zcta5", score_col:f"hpsa_{label}_score"})
        agg = agg.merge(sc, on="zcta5", how="left")
    return agg


def load_hrsa():
    out = _hrsa_one("hpsa_primary_care.csv", "primary_care").merge(
        _hrsa_one("hpsa_mental_health.csv", "mental_health"),
        on="zcta5", how="outer")
    for c in out.columns:
        if c.endswith("_flag"):
            out[c] = out[c].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------------
# Census ACS — state-by-state tract JSON files
# ---------------------------------------------------------------------------
def load_acs_tract():
    acs_dir = DATA_ROOT / "census_acs"
    files = sorted(acs_dir.glob("acs5_2022_tract_state*.json"))
    if not files:
        # also handle the typo'd extension the user mentioned
        files = sorted(acs_dir.glob("acs5_2022_tract_state*.josn"))
    if not files:
        print("[warn] No ACS state JSON files found; skipping ACS layer.")
        return pd.DataFrame()

    frames = []
    for fp in files:
        with open(fp) as f:
            raw = json.load(f)
        header, *rows = raw
        df = pd.DataFrame(rows, columns=header)
        # Standard Census API columns: variables + 'state','county','tract'
        for var in ACS_VARS.keys():
            if var in df.columns:
                df[ACS_VARS[var]] = _safe_numeric(df[var])
        df["tract_fips"] = (
            df["state"].astype(str).str.zfill(2) +
            df["county"].astype(str).str.zfill(3) +
            df["tract"].astype(str).str.zfill(6)
        )
        keep = ["tract_fips"] + [v for v in ACS_VARS.values() if v in df.columns]
        frames.append(df[keep])

    out = pd.concat(frames, ignore_index=True)

    # Derived columns — guard every reference so missing ACS vars don't crash.
    if "pop_male_65_plus" in out.columns and "pop_female_65_plus" in out.columns:
        out["pop_65_plus"] = (out["pop_male_65_plus"].fillna(0) +
                              out["pop_female_65_plus"].fillna(0))
    elif "pop_male_65_plus" in out.columns:
        out["pop_65_plus"] = out["pop_male_65_plus"].fillna(0)
    elif "pop_female_65_plus" in out.columns:
        out["pop_65_plus"] = out["pop_female_65_plus"].fillna(0)

    if "total_population" in out.columns:
        for race_col in ["pop_black", "pop_hispanic", "pop_asian"]:
            if race_col in out.columns:
                out[f"pct_{race_col.replace('pop_','')}"] = (
                    out[race_col] / out["total_population"].replace(0, np.nan)
                )
        if "pop_65_plus" in out.columns:
            out["pct_65_plus"] = (
                out["pop_65_plus"] / out["total_population"].replace(0, np.nan)
            )

    out["county_fips"] = out["tract_fips"].str[:5]
    out["state_fips"] = out["tract_fips"].str[:2]
    print(f"      ACS columns available: "
          f"{[c for c in out.columns if c not in ('tract_fips','county_fips','state_fips')]}")
    return out


# ---------------------------------------------------------------------------
# CDC SVI — tract and county
# ---------------------------------------------------------------------------
def _find_svi_file(subdir):
    """SVI files are typically named like SVI_2020_US.csv or similar."""
    base = DATA_ROOT / "cdc_svi"
    # Try mirror first, then top-level
    candidates = [
        base / "_lpiep_cdc_svi_mirror" / "csv" / subdir,
        base / subdir,
    ]
    for c in candidates:
        if c.exists():
            csvs = list(c.glob("*.csv"))
            if csvs:
                return csvs[0]
    return None


SVI_THEMES = {
    "RPL_THEMES": "svi_overall",
    "RPL_THEME1": "svi_socioeconomic",
    "RPL_THEME2": "svi_household",
    "RPL_THEME3": "svi_minority_lang",
    "RPL_THEME4": "svi_housing_transport",
}


def load_svi_tract():
    fp = _find_svi_file("tract")
    if not fp:
        print("[warn] No SVI tract file found.")
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    fips_col = next((c for c in ["FIPS","TRACT_FIPS","GEOID"] if c in df.columns), None)
    if fips_col is None:
        print(f"[warn] SVI tract: no FIPS column. Cols: {df.columns.tolist()[:15]}")
        return pd.DataFrame()
    df["tract_fips"] = df[fips_col].astype(str).str.zfill(11)
    keep = ["tract_fips"] + [SVI_THEMES[k] for k in SVI_THEMES if k in df.columns]
    for src, dst in SVI_THEMES.items():
        if src in df.columns:
            df[dst] = _safe_numeric(df[src])
            # SVI uses -999 for missing
            df.loc[df[dst] < 0, dst] = np.nan
    return df[["tract_fips"] + [v for v in SVI_THEMES.values() if v in df.columns]]


def load_svi_county():
    fp = _find_svi_file("county")
    if not fp:
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    fips_col = next((c for c in ["FIPS","COUNTY_FIPS","STCNTY"] if c in df.columns), None)
    state_col = next((c for c in df.columns if c.upper()=="ST_ABBR" or c.upper()=="STATE"), None)
    if fips_col is None:
        return pd.DataFrame()
    df["county_fips"] = df[fips_col].astype(str).str.zfill(5)
    for src, dst in SVI_THEMES.items():
        if src in df.columns:
            df[dst] = _safe_numeric(df[src])
            df.loc[df[dst] < 0, dst] = np.nan
    keep_cols = ["county_fips"] + [v for v in SVI_THEMES.values() if v in df.columns]
    if state_col: keep_cols.append(state_col)
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build():
    print("[1/6] CMS Hospital Compare...")
    cms = load_cms_hospitals()
    print(f"      {len(cms):,} hospitals")

    print("[2/6] CDC PLACES ZCTA...")
    places_zcta = load_places_zcta()

    print("[3/6] HRSA shortage areas...")
    hrsa = load_hrsa()

    print("[4/6] Census ACS tract...")
    acs_tract = load_acs_tract()
    print(f"      {len(acs_tract):,} tracts")

    print("[5/6] CDC SVI tract + county...")
    svi_tract = load_svi_tract()
    svi_county = load_svi_county()
    print(f"      {len(svi_tract):,} tracts, {len(svi_county):,} counties")

    print("[6/6] PLACES tract...")
    places_tract = load_places_tract()

    # --- Primary: ZCTA-keyed hospital risk matrix ---
    risk = cms.merge(places_zcta, on="zcta5", how="left")
    risk = risk.merge(hrsa, on="zcta5", how="left")
    for c in ["hpsa_primary_care_flag","hpsa_mental_health_flag"]:
        if c in risk.columns: risk[c] = risk[c].fillna(0).astype(int)

    # Bridge: state-level SVI average → join to hospitals by state
    if not svi_county.empty:
        svi_county["state_fips"] = svi_county["county_fips"].str[:2]
        svi_cols_present = [c for c in SVI_THEMES.values() if c in svi_county.columns]
        if svi_cols_present:
            agg_spec = {f"state_avg_{c}": (c, "mean") for c in svi_cols_present}
            state_svi = svi_county.groupby("state_fips").agg(**agg_spec).reset_index()
            state_svi["state_abbr"] = state_svi["state_fips"].map(STATE_FIPS_TO_ABBR)
            risk = risk.merge(state_svi.drop(columns="state_fips"),
                              left_on="State", right_on="state_abbr", how="left")
        else:
            print(f"[warn] SVI county loaded but no RPL_THEME* columns found. "
                  f"Cols: {svi_county.columns.tolist()[:20]}")
            print("       State-level SVI join skipped. Tract SVI still works.")

    # --- Secondary: tract-keyed equity table ---
    tract_eq = places_tract.copy()
    if not acs_tract.empty:
        tract_eq = tract_eq.merge(acs_tract, on=["tract_fips","county_fips","state_fips"],
                                  how="left")
    if not svi_tract.empty:
        tract_eq = tract_eq.merge(svi_tract, on="tract_fips", how="left")

    # --- County rollup for UI / state-level views ---
    county_eq = svi_county.copy() if not svi_county.empty else pd.DataFrame()

    risk.to_parquet(OUT_DIR / "hospital_risk_matrix.parquet", index=False)
    tract_eq.to_parquet(OUT_DIR / "tract_equity.parquet", index=False)
    if not county_eq.empty:
        county_eq.to_parquet(OUT_DIR / "county_equity.parquet", index=False)

    print("\nOutputs:")
    print(f"  hospital_risk_matrix.parquet   {risk.shape}")
    print(f"  tract_equity.parquet           {tract_eq.shape}")
    if not county_eq.empty:
        print(f"  county_equity.parquet          {county_eq.shape}")

    print("\nCoverage:")
    print(f"  hospitals w/ PLACES data:        {risk['diabetes_prev'].notna().sum():,}")
    print(f"  hospitals w/ state SVI:          {risk.get('state_avg_svi_overall', pd.Series()).notna().sum():,}")
    print(f"  tracts w/ ACS pop:               {tract_eq.get('total_population', pd.Series()).notna().sum():,}")
    print(f"  tracts w/ SVI overall:           {tract_eq.get('svi_overall', pd.Series()).notna().sum():,}")


if __name__ == "__main__":
    build()