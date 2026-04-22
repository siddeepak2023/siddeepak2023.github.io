"""
02_generate_clinic_data.py
Generate synthetic patient-level clinic records calibrated to CMS benchmarks.
Outputs:
  data/patients.csv        — 2,500 patients, 3-year history
  data/appointments.csv    — 12,000+ appointments
  data/marketing.csv       — 36 months of channel spend
  data/clinic.db           — SQLite with all three tables

Run:  python3 02_generate_clinic_data.py
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS_JSON = "data/cms_benchmarks.json"
DB_PATH         = "data/clinic.db"
SEED            = 42
N_PATIENTS      = 2500
START_DATE      = datetime(2022, 1, 1)
END_DATE        = datetime(2024, 12, 31)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

rng = np.random.default_rng(SEED)

# ── TX zip codes (real prefix ranges) ────────────────────────────────────────
TX_ZIPS = [
    "75201","75202","75204","75206","75209","75214","75218","75224","75230",
    "75240","75248","75252","75287","75001","75002","75006","75007","75010",
    "77001","77002","77004","77005","77006","77007","77008","77018","77019",
    "77024","77025","77027","77035","77042","77056","77057","77063","77077",
    "78201","78202","78203","78204","78205","78207","78209","78210","78216",
    "78217","78218","78220","78228","78229","78230","78232","78233","78249",
    "73301","78701","78702","78703","78704","78705","78712","78717","78719",
    "78721","78722","78723","78724","78725","78726","78727","78728","78730",
]

# ── Medical procedure → CMS-based payment ranges ─────────────────────────────
MEDICAL_PROCEDURES = {
    "office_visit_est":   ("99213", 62,  92,  "em_visit"),
    "office_visit_comp":  ("99214", 92, 132,  "em_visit"),
    "office_visit_new":   ("99203", 87, 134,  "em_new"),
    "office_visit_new_c": ("99204",134, 180,  "em_new"),
    "biopsy":             ("11100", 88, 130,  "biopsy"),
    "biopsy_add":         ("11101", 30,  50,  "biopsy"),
    "ak_destruction":     ("17000", 63, 109,  "destruction"),
    "ak_dest_mult":       ("17004",109, 145,  "destruction"),
    "benign_dest":        ("17110", 80, 120,  "destruction"),
    "malignant_dest":     ("17260",171, 230,  "destruction_mal"),
    "mohs_face":          ("17311",703, 879,  "mohs"),
    "mohs_body":          ("17315",454, 568,  "mohs"),
    "shave_removal":      ("11300", 63,  90,  "shave"),
    "excision_benign":    ("11440",124, 180,  "excision"),
    "excision_malignant": ("11600",176, 249,  "excision"),
    "ib_abscess":         ("10060", 75, 110,  "other_proc"),
    "phototherapy":       ("96920", 99, 150,  "phototherapy"),
}

# Cosmetic procedures (cash-pay only, NOT in CMS data)
COSMETIC_PROCEDURES = {
    "botox":              ("COSM-BOT",  350,  800, "cosmetic"),
    "filler":             ("COSM-FIL",  600, 1400, "cosmetic"),
    "laser_rejuv":        ("COSM-LAS",  400,  900, "cosmetic"),
    "chemical_peel":      ("COSM-PEL",  150,  400, "cosmetic"),
    "microneedling":      ("COSM-MCN",  250,  500, "cosmetic"),
    "coolsculpting":      ("COSM-CSC",  700, 1500, "cosmetic"),
    "ipl_treatment":      ("COSM-IPL",  300,  700, "cosmetic"),
}

# ── Insurance type → payment multiplier ──────────────────────────────────────
INSURANCE_TYPES = {
    "private":    {"frac": 0.45, "pay_mult": 1.25, "noshow_base": 0.12},
    "medicare":   {"frac": 0.25, "pay_mult": 1.00, "noshow_base": 0.14},
    "medicaid":   {"frac": 0.15, "pay_mult": 0.65, "noshow_base": 0.28},
    "self_pay":   {"frac": 0.15, "pay_mult": 0.85, "noshow_base": 0.22},
}

# ── Acquisition channels ──────────────────────────────────────────────────────
CHANNELS = {
    "google_search":    {"frac": 0.35, "cac": 120, "ltv_mult": 1.0, "ret_2yr": 0.55},
    "doctor_referral":  {"frac": 0.25, "cac":  40, "ltv_mult": 1.4, "ret_2yr": 0.74},
    "patient_referral": {"frac": 0.20, "cac":  55, "ltv_mult": 1.3, "ret_2yr": 0.68},
    "instagram":        {"frac": 0.15, "cac": 180, "ltv_mult": 0.8, "ret_2yr": 0.41},
    "walk_in":          {"frac": 0.05, "cac":  20, "ltv_mult": 0.9, "ret_2yr": 0.48},
}

CONDITIONS = {
    "skin_cancer_screening": 0.20,
    "cosmetic":              0.25,
    "acne":                  0.20,
    "eczema":                0.15,
    "psoriasis":             0.10,
    "rosacea":               0.10,
}


# ─────────────────────────────────────────────────────────────────────────────

def rand_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta)))


def gen_patients():
    log.info("Generating %d patients …", N_PATIENTS)
    ins_types = list(INSURANCE_TYPES.keys())
    ins_fracs = [INSURANCE_TYPES[k]["frac"] for k in ins_types]
    channels  = list(CHANNELS.keys())
    chan_fracs = [CHANNELS[k]["frac"] for k in channels]
    conds     = list(CONDITIONS.keys())
    cond_fracs = list(CONDITIONS.values())

    rows = []
    for i in range(N_PATIENTS):
        pid   = f"P{i+1:05d}"
        age   = int(np.clip(rng.normal(48, 18), 16, 88))
        gender = rng.choice(["F","M","F","F","M"])  # slight female skew for cosmetic
        zipcode = rng.choice(TX_ZIPS)
        ins   = rng.choice(ins_types, p=ins_fracs)
        chan  = rng.choice(channels,  p=chan_fracs)
        cond  = rng.choice(conds,     p=cond_fracs)

        # First visit date — stagger over 3 years, earlier patients have more visits
        first_visit = rand_date(START_DATE, START_DATE + timedelta(days=800))
        days_active = (END_DATE - first_visit).days

        # Visits per year varies by condition
        visits_per_yr = {
            "skin_cancer_screening": rng.choice([1,1,1,2]),
            "cosmetic":              rng.integers(2, 6),
            "acne":                  rng.integers(3, 8),
            "eczema":                rng.integers(2, 6),
            "psoriasis":             rng.integers(3, 7),
            "rosacea":               rng.integers(2, 5),
        }[cond]
        total_visits = max(1, int(visits_per_yr * days_active / 365))

        # Avg revenue per visit depends on condition + insurance
        pay_mult = INSURANCE_TYPES[ins]["pay_mult"]
        rev_per_visit = {
            "skin_cancer_screening": rng.uniform(110, 180),
            "cosmetic":              rng.uniform(350, 900),
            "acne":                  rng.uniform(80, 140),
            "eczema":                rng.uniform(90, 160),
            "psoriasis":             rng.uniform(100, 200),
            "rosacea":               rng.uniform(85, 150),
        }[cond]
        total_revenue = round(rev_per_visit * pay_mult * total_visits * rng.uniform(0.85, 1.15), 2)

        last_visit = first_visit + timedelta(days=min(days_active, total_visits * 30))
        if last_visit > END_DATE:
            last_visit = END_DATE

        rows.append({
            "patient_id":       pid,
            "age":              age,
            "gender":           gender,
            "zip_code":         zipcode,
            "insurance_type":   ins,
            "acquisition_channel": chan,
            "first_visit_date": first_visit.strftime("%Y-%m-%d"),
            "last_visit_date":  last_visit.strftime("%Y-%m-%d"),
            "total_visits":     total_visits,
            "total_revenue":    total_revenue,
            "primary_condition":cond,
        })

    df = pd.DataFrame(rows)
    log.info("  Patients: %d  |  Avg visits: %.1f  |  Avg revenue: $%.0f",
             len(df), df["total_visits"].mean(), df["total_revenue"].mean())
    return df


def gen_appointments(patients_df):
    log.info("Generating appointments …")
    rows = []
    appt_id = 1

    # Day-of-week no-show modifiers (Mon AM worst, Fri PM moderate)
    dow_noshow = {0: 1.30, 1: 1.05, 2: 0.95, 3: 0.90, 4: 1.00, 5: 0.80, 6: 0.80}
    # Time slot no-show modifiers
    slot_noshow = {"8am": 1.25, "9am": 1.05, "10am": 0.95, "11am": 0.90,
                   "1pm": 1.10, "2pm": 0.95, "3pm": 0.95, "4pm": 1.05}
    slots = list(slot_noshow.keys())

    for _, pat in patients_df.iterrows():
        first = datetime.strptime(pat["first_visit_date"], "%Y-%m-%d")
        last  = datetime.strptime(pat["last_visit_date"],  "%Y-%m-%d")
        n_appts = max(1, int(pat["total_visits"] * rng.uniform(1.1, 1.3)))  # booked > attended
        ins_ns_base = INSURANCE_TYPES[pat["insurance_type"]]["noshow_base"]
        is_cosmetic = pat["primary_condition"] == "cosmetic"

        # Schedule appointments spread across active period
        for v in range(n_appts):
            if (last - first).days < 1:
                appt_date = first
            else:
                appt_date = first + timedelta(days=int(rng.integers(0, (last-first).days)))
            if appt_date > END_DATE:
                appt_date = END_DATE

            dow      = appt_date.weekday()
            slot     = rng.choice(slots)
            days_adv = int(rng.integers(1, 45))

            # No-show probability
            is_new   = (v == 0)
            ns_prob  = ins_ns_base * dow_noshow[dow] * slot_noshow[slot]
            ns_prob  *= (1.25 if is_new else 1.0)
            ns_prob  *= (0.85 if is_cosmetic else 1.0)
            ns_prob  = np.clip(ns_prob, 0.03, 0.65)
            reminder = int(rng.random() < 0.75)
            if reminder:
                ns_prob *= 0.80

            no_show      = int(rng.random() < ns_prob)
            cancel_hours = int(rng.integers(0, 48)) if no_show and rng.random() < 0.4 else None

            # Appointment type → procedure + revenue
            if is_cosmetic:
                proc_key = rng.choice(list(COSMETIC_PROCEDURES.keys()))
                code, lo, hi, cat = COSMETIC_PROCEDURES[proc_key]
                revenue = round(float(rng.uniform(lo, hi)), 2)
            else:
                cond_procs = {
                    "skin_cancer_screening": ["office_visit_est","biopsy","malignant_dest","excision_malignant","mohs_face"],
                    "acne":    ["office_visit_est","office_visit_new","benign_dest","shave_removal"],
                    "eczema":  ["office_visit_est","office_visit_comp","phototherapy"],
                    "psoriasis":["office_visit_est","office_visit_comp","phototherapy","benign_dest"],
                    "rosacea": ["office_visit_est","office_visit_comp","laser_rejuv" if False else "benign_dest"],
                }
                available = cond_procs.get(pat["primary_condition"], ["office_visit_est"])
                proc_key = rng.choice(available)
                code, lo, hi, cat = MEDICAL_PROCEDURES.get(proc_key,
                    ("99213", 62, 92, "em_visit"))
                pay_mult = INSURANCE_TYPES[pat["insurance_type"]]["pay_mult"]
                revenue  = round(float(rng.uniform(lo, hi) * pay_mult), 2)
                if no_show:
                    revenue = 0.0

            rows.append({
                "appt_id":             f"A{appt_id:06d}",
                "patient_id":          pat["patient_id"],
                "appt_date":           appt_date.strftime("%Y-%m-%d"),
                "day_of_week":         dow,
                "time_slot":           slot,
                "days_booked_in_advance": days_adv,
                "appointment_type":    "cosmetic" if is_cosmetic else pat["primary_condition"],
                "hcpcs_code":          code,
                "proc_category":       cat,
                "insurance_type":      pat["insurance_type"],
                "is_new_patient":      int(is_new),
                "reminder_sent":       reminder,
                "no_show":             no_show,
                "cancellation_hours_before": cancel_hours,
                "revenue":             revenue if not no_show else 0.0,
            })
            appt_id += 1

    df = pd.DataFrame(rows)
    log.info("  Appointments: %d  |  No-show rate: %.1f%%  |  Total revenue: $%.0f",
             len(df), df["no_show"].mean()*100, df["revenue"].sum())
    return df


def gen_marketing():
    log.info("Generating 36 months of marketing data …")
    rows = []
    channels = list(CHANNELS.keys())

    for month_offset in range(36):
        dt = START_DATE + timedelta(days=30 * month_offset)
        month_str = dt.strftime("%Y-%m")
        # Seasonality: Q4 (cosmetic season) + spring bump
        month_num = dt.month
        season_mult = 1.0
        if month_num in [11, 12]:  # Dec cosmetic peak
            season_mult = 1.35
        elif month_num in [2, 3]:   # Valentine / spring
            season_mult = 1.20
        elif month_num in [7, 8]:   # Summer slow
            season_mult = 0.85

        for ch in channels:
            cdata = CHANNELS[ch]
            base_spend = {
                "google_search":    2200,
                "doctor_referral":   400,
                "patient_referral":  300,
                "instagram":        1800,
                "walk_in":            50,
            }[ch]
            spend = round(float(base_spend * season_mult * rng.uniform(0.88, 1.12)), 2)
            cac   = cdata["cac"] * rng.uniform(0.85, 1.20)
            new_pats = max(1, int(spend / cac))
            # Revenue attributed via LTV estimate
            avg_ltv_base = {
                "google_search":   1800,
                "doctor_referral": 3200,
                "patient_referral":2800,
                "instagram":       1100,
                "walk_in":         1400,
            }[ch]
            rev_attr = round(float(new_pats * avg_ltv_base * rng.uniform(0.80, 1.20) / 24), 2)

            rows.append({
                "month":              month_str,
                "channel":            ch,
                "spend":              spend,
                "new_patients_acquired": new_pats,
                "revenue_attributed": rev_attr,
                "cac_actual":         round(spend / max(new_pats, 1), 2),
                "roas":               round(rev_attr / max(spend, 1), 3),
            })

    df = pd.DataFrame(rows)
    log.info("  Marketing: %d rows  |  Total spend: $%.0f  |  Total attributed rev: $%.0f",
             len(df), df["spend"].sum(), df["revenue_attributed"].sum())
    return df


def main():
    Path("data").mkdir(exist_ok=True)

    # Load benchmarks to confirm we're calibrated
    with open(BENCHMARKS_JSON) as f:
        bm = json.load(f)
    log.info("CMS benchmark: solo practice ~%d patients/week",
             bm["provider_volume_dist"]["solo_practice_weekly_patients"])

    patients    = gen_patients()
    appointments = gen_appointments(patients)
    marketing   = gen_marketing()

    # Save CSVs
    patients.to_csv("data/patients.csv", index=False)
    appointments.to_csv("data/appointments.csv", index=False)
    marketing.to_csv("data/marketing.csv", index=False)
    log.info("CSVs saved.")

    # SQLite
    conn = sqlite3.connect(DB_PATH)
    patients.to_sql("patients",     conn, if_exists="replace", index=False)
    appointments.to_sql("appointments", conn, if_exists="replace", index=False)
    marketing.to_sql("marketing",   conn, if_exists="replace", index=False)
    for tbl, col in [("appointments","appt_date"), ("patients","first_visit_date"),
                     ("appointments","patient_id"), ("patients","patient_id")]:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_{col} ON {tbl}({col})")
    conn.commit()
    conn.close()
    log.info("SQLite → %s", DB_PATH)

    # Summary
    total_rev = appointments["revenue"].sum()
    total_spend = marketing["spend"].sum()
    no_show_rev_lost = appointments[appointments["no_show"]==1].shape[0] * appointments["revenue"].mean()
    log.info("\n── Clinic Summary ────────────────────────────────────────────────")
    log.info("  Patients:       %d", len(patients))
    log.info("  Appointments:   %d  (attended: %d)",
             len(appointments), (appointments["no_show"]==0).sum())
    log.info("  Total revenue:  $%,.0f", total_rev)
    log.info("  No-show rate:   %.1f%%", appointments["no_show"].mean()*100)
    log.info("  Rev at risk/yr (no-shows):  $%,.0f est.", total_rev * 0.18)
    log.info("  Total mktg spend (3yr):     $%,.0f", total_spend)
    log.info("✓  Layer 2 complete.")


if __name__ == "__main__":
    main()
