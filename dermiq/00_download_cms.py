"""
00_download_cms.py
Download CMS Medicare Physician & Other Practitioners data filtered to Dermatology.
Saves data/cms_dermatology.csv.

If network download is blocked, creates a realistic 500-row sample and prints
manual download instructions.

Run:  python3 00_download_cms.py
"""

import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

Path("data").mkdir(exist_ok=True)
OUTPUT_CSV = "data/cms_dermatology.csv"

# ── CMS column schema ────────────────────────────────────────────────────────
CMS_COLUMNS = [
    "Rndrng_NPI", "Rndrng_Prvdr_Last_Org_Name", "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_City", "Rndrng_Prvdr_State_Abrvtn", "Rndrng_Prvdr_Zip5",
    "Rndrng_Prvdr_Type", "HCPCS_Cd", "HCPCS_Desc",
    "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Allowed_Amt", "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]

# ── Real dermatology procedure codes with CMS-benchmarked payments ────────────
DERM_PROCEDURES = {
    # (HCPCS_Cd, HCPCS_Desc, avg_submitted, avg_allowed, avg_payment, avg_stdz)
    "99213": ("Office/outpatient visit, established pt, low complexity",        140,  78,  62,  66),
    "99214": ("Office/outpatient visit, established pt, moderate complexity",   215, 115,  92,  98),
    "99212": ("Office/outpatient visit, established pt, straightforward",        95,  47,  37,  40),
    "99215": ("Office/outpatient visit, established pt, high complexity",       310, 165, 132, 140),
    "99203": ("Office/outpatient visit, new pt, low complexity",                180, 109,  87,  93),
    "99204": ("Office/outpatient visit, new pt, moderate complexity",           280, 167, 134, 142),
    "11100": ("Biopsy of skin lesion, single",                                  260, 110,  88,  92),
    "11101": ("Biopsy of skin lesion, each additional",                          85,  37,  30,  31),
    "17000": ("Destruction actinic keratosis, first lesion",                    180,  79,  63,  67),
    "17003": ("Destruction actinic keratosis, 2-14 lesions each",                35,  10,   8,   9),
    "17004": ("Destruction actinic keratosis, 15+ lesions",                     280, 136, 109, 115),
    "17110": ("Destruction benign lesions, 1-14 lesions",                       230, 100,  80,  85),
    "17260": ("Destruction malignant lesion, trunk/arm/leg, ≤0.5cm",            480, 214, 171, 181),
    "17261": ("Destruction malignant lesion, trunk/arm/leg, 0.6-1.0cm",         530, 230, 184, 195),
    "17311": ("Mohs micrographic surgery, face/head/neck, 1st stage",          1900, 879, 703, 745),
    "17312": ("Mohs micrographic surgery, face/head/neck, each add'l stage",    970, 412, 330, 349),
    "17315": ("Mohs micrographic surgery, trunk/arm/leg, 1st stage",           1500, 568, 454, 481),
    "11300": ("Shave removal skin lesion, face/head/neck, ≤0.5cm",              190,  79,  63,  67),
    "11301": ("Shave removal skin lesion, face/head/neck, 0.6-1.0cm",           220,  86,  69,  73),
    "10060": ("Incision and drainage abscess, simple",                          220,  94,  75,  79),
    "96920": ("Laser treatment for inflammatory skin, <250 sq cm",              420, 124,  99, 105),
    "11720": ("Debridement of nail(s), 1-5",                                     65,  27,  22,  23),
    "99211": ("Office/outpatient visit, minimal complexity",                     55,  21,  17,  18),
    "17020": ("Destruction of lesion, premalignant, local",                     150,  68,  54,  57),
    "11440": ("Excision benign lesion, face/ear/lid, ≤0.5cm",                   380, 155, 124, 131),
    "11441": ("Excision benign lesion, face/ear/lid, 0.6-1.0cm",                450, 183, 146, 155),
    "11600": ("Excision malignant lesion, trunk/arm/leg, ≤0.5cm",               540, 220, 176, 186),
    "11601": ("Excision malignant lesion, trunk/arm/leg, 0.6-1.0cm",            610, 249, 199, 211),
    "96910": ("Photochemotherapy, tars and/or UV-B",                             170,  65,  52,  55),
}

# ── Provider archetypes (realistic for synthetic sample) ─────────────────────
STATES_WEIGHTS = {
    "TX": 0.14, "CA": 0.12, "FL": 0.09, "NY": 0.08, "IL": 0.05,
    "PA": 0.05, "OH": 0.04, "GA": 0.04, "NC": 0.04, "AZ": 0.03,
    "VA": 0.03, "MA": 0.03, "WA": 0.03, "CO": 0.03, "TN": 0.03,
    "MN": 0.02, "MD": 0.02, "NJ": 0.02, "SC": 0.02, "MO": 0.02,
    "OTHER": 0.06,
}

TX_CITIES = ["Dallas", "Houston", "Austin", "San Antonio", "Fort Worth",
             "Plano", "Frisco", "The Woodlands", "Sugar Land", "McKinney"]
OTHER_CITIES = ["Chicago", "Los Angeles", "New York", "Miami", "Atlanta",
                "Phoenix", "Seattle", "Boston", "Denver", "Nashville"]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White",
    "Harris", "Martin", "Thompson", "Young", "Lee", "Walker", "Hall",
    "Allen", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson",
    "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Evans",
    "Patel", "Kim", "Chen", "Rodriguez", "Martinez",
]
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa", "Christopher",
    "Emily", "Daniel", "Ashley", "Mark", "Stephanie", "Jennifer", "Andrew",
]


def try_cms_api_download():
    """Attempt to download CMS data via API with dermatology filter."""
    # CMS Data Portal API endpoints (try in order)
    endpoints = [
        # Socrata/DKAN API with filter
        "https://data.cms.gov/data-api/v1/dataset/9552359e-0b03-4ed4-acb6-6fce3792e232/data"
        "?filter[Rndrng_Prvdr_Type]=Dermatology&size=50000",
        # Direct CSV with API query (2022 dataset)
        "https://data.cms.gov/resource/fs4p-t5eq.csv"
        "?Rndrng_Prvdr_Type=Dermatology&$limit=50000",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/csv",
    }
    for url in endpoints:
        try:
            log.info("Trying: %s …", url[:80])
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200 and len(r.content) > 10000:
                log.info("  Downloaded %d bytes", len(r.content))
                if url.endswith(".csv") or "csv" in r.headers.get("Content-Type",""):
                    df = pd.read_csv(io.StringIO(r.text))
                else:
                    data = r.json()
                    if isinstance(data, list) and data:
                        df = pd.DataFrame(data)
                    else:
                        continue
                # Filter to dermatology
                type_col = next((c for c in df.columns if "type" in c.lower()), None)
                if type_col:
                    df = df[df[type_col].str.lower().str.contains("derm", na=False)]
                if len(df) > 100:
                    log.info("  Filtered to %d dermatology rows", len(df))
                    return df
        except Exception as e:
            log.warning("  Failed: %s", e)
    return None


def build_realistic_sample(n=500, seed=42):
    """
    Build a 500-row realistic CMS sample calibrated to published 2022 statistics.
    Each row = one provider × one procedure code (same structure as real CMS file).
    """
    log.info("Building realistic CMS sample (%d rows) …", n)
    rng = np.random.default_rng(seed)

    proc_codes = list(DERM_PROCEDURES.keys())
    # Weight procedures by approximate national volume
    proc_weights = [
        0.18, 0.14, 0.10, 0.04, 0.05, 0.04,  # E/M codes 99213-99204
        0.06, 0.03, 0.05, 0.02, 0.02,          # biopsies + AK destruction
        0.03, 0.02, 0.02, 0.02, 0.01, 0.01,    # malignant + Mohs
        0.02, 0.01, 0.02, 0.01, 0.01,          # shave, I&D, laser, nails
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # excisions
    ]
    # Normalise weights to sum to 1
    proc_weights = np.array(proc_weights[:len(proc_codes)])
    proc_weights /= proc_weights.sum()

    # Generate ~35 distinct providers
    n_providers = min(35, n // 12)
    npis = rng.integers(1_000_000_000, 2_000_000_000, size=n_providers)
    last_names = rng.choice(LAST_NAMES, n_providers, replace=False)
    first_names = rng.choice(FIRST_NAMES, n_providers)

    states = list(STATES_WEIGHTS.keys())
    state_probs = list(STATES_WEIGHTS.values())
    prov_states = []
    prov_cities = []
    for _ in range(n_providers):
        s = rng.choice(states, p=np.array(state_probs)/sum(state_probs))
        if s == "TX":
            c = rng.choice(TX_CITIES)
        elif s == "OTHER":
            s = rng.choice(["WI", "LA", "AL", "KS", "NE"])
            c = rng.choice(OTHER_CITIES)
        else:
            c = rng.choice(OTHER_CITIES)
        prov_states.append(s)
        prov_cities.append(c)

    rows = []
    for i in range(n):
        prov_idx = rng.integers(0, n_providers)
        code = rng.choice(proc_codes, p=proc_weights)
        desc, sbmt, allowed, payment, stdz = DERM_PROCEDURES[code]

        # Add realistic variation (±20%)
        noise = rng.normal(1.0, 0.12)
        tot_benes = int(rng.integers(11, 180))
        tot_srvcs = int(tot_benes * rng.uniform(1.0, 1.8))

        rows.append({
            "Rndrng_NPI":                  str(npis[prov_idx]),
            "Rndrng_Prvdr_Last_Org_Name":  last_names[prov_idx],
            "Rndrng_Prvdr_First_Name":     first_names[prov_idx],
            "Rndrng_Prvdr_City":           prov_cities[prov_idx],
            "Rndrng_Prvdr_State_Abrvtn":   prov_states[prov_idx],
            "Rndrng_Prvdr_Zip5":           str(rng.integers(70000, 79999)) if prov_states[prov_idx] == "TX" else str(rng.integers(10000, 99999)),
            "Rndrng_Prvdr_Type":           "Dermatology",
            "HCPCS_Cd":                    code,
            "HCPCS_Desc":                  desc,
            "Tot_Benes":                   tot_benes,
            "Tot_Srvcs":                   tot_srvcs,
            "Tot_Bene_Day_Srvcs":          tot_srvcs,
            "Avg_Sbmtd_Chrg":              round(sbmt  * noise, 2),
            "Avg_Mdcr_Allowed_Amt":        round(allowed * noise, 2),
            "Avg_Mdcr_Pymt_Amt":           round(payment * noise, 2),
            "Avg_Mdcr_Stdzd_Amt":          round(stdz * noise, 2),
        })

    return pd.DataFrame(rows)


def print_download_instructions():
    log.info("")
    log.info("=" * 70)
    log.info("MANUAL DOWNLOAD INSTRUCTIONS")
    log.info("=" * 70)
    log.info("URL: https://data.cms.gov/provider-summary-by-type-of-service/")
    log.info("     medicare-physician-other-practitioners/")
    log.info("     medicare-physician-other-practitioners-by-provider-and-service")
    log.info("")
    log.info("Steps:")
    log.info("  1. Go to the URL above")
    log.info("  2. Click 'Download' → CSV")
    log.info("  3. The file is large (~2GB). Filter in Excel/Python:")
    log.info("     df = pd.read_csv('file.csv')")
    log.info("     df = df[df['Rndrng_Prvdr_Type']=='Dermatology']")
    log.info("     df.to_csv('data/cms_dermatology.csv', index=False)")
    log.info("  OR use the API with offset pagination:")
    log.info("  https://data.cms.gov/resource/fs4p-t5eq.json")
    log.info("     ?Rndrng_Prvdr_Type=Dermatology&$limit=50000&$offset=0")
    log.info("=" * 70)
    log.info("")


def summarize(df):
    log.info("")
    log.info("── CMS Dermatology Summary ──────────────────────────────────────")
    log.info("  Total rows:          %d", len(df))
    log.info("  Unique providers:    %d", df["Rndrng_NPI"].nunique())
    log.info("  States covered:      %d", df["Rndrng_Prvdr_State_Abrvtn"].nunique())

    # Top 20 procedures by total services
    top_procs = (
        df.groupby(["HCPCS_Cd", "HCPCS_Desc"])["Tot_Srvcs"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    log.info("\n  Top 20 Procedures by National Volume:")
    log.info("  %-8s %-48s %12s %12s", "Code", "Description", "Tot Services", "Avg Payment")
    for _, r in top_procs.iterrows():
        avg_pay = df[df["HCPCS_Cd"] == r["HCPCS_Cd"]]["Avg_Mdcr_Pymt_Amt"].mean()
        log.info("  %-8s %-48s %12,.0f  $%10.2f",
                 r["HCPCS_Cd"], r["HCPCS_Desc"][:46], r["Tot_Srvcs"], avg_pay)

    log.info("\n  Avg Medicare payment by procedure (top 10):")
    pay_by_proc = df.groupby("HCPCS_Cd")["Avg_Mdcr_Pymt_Amt"].mean().sort_values(ascending=False).head(10)
    for code, pay in pay_by_proc.items():
        desc = df[df["HCPCS_Cd"] == code]["HCPCS_Desc"].iloc[0][:50]
        log.info("  %-8s $%7.2f  %s", code, pay, desc)


def main():
    # Try real download
    log.info("Attempting CMS API download …")
    df = try_cms_api_download()

    if df is not None and len(df) > 100:
        log.info("✓ Downloaded real CMS data: %d rows", len(df))
        # Standardise column names
        col_map = {c: c for c in df.columns}
        df = df.rename(columns=col_map)
        source = "REAL CMS DATA"
    else:
        log.warning("CMS download unavailable — building realistic sample.")
        print_download_instructions()
        df = build_realistic_sample(n=500)
        source = "SYNTHETIC SAMPLE (replace with real CMS file per instructions above)"

    df.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved → %s  [%s]", OUTPUT_CSV, source)

    # Write source flag so downstream scripts know
    with open("data/cms_source.txt", "w") as f:
        f.write(source)

    summarize(df)
    log.info("\n✓  Layer 0 complete.")


if __name__ == "__main__":
    main()
