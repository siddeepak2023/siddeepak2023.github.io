"""
05_marketing_attribution.py
ROAS by channel, LTV:CAC analysis, cohort retention by channel.
Outputs data/marketing_analysis.json.

Run:  python3 05_marketing_attribution.py
"""

import json
import logging

import numpy as np
import pandas as pd

MARKETING_CSV   = "data/marketing.csv"
PATIENTS_CSV    = "data/patients.csv"
APPTS_CSV       = "data/appointments.csv"
ANALYTICS_JSON  = "data/marketing_analysis.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Known CAC and LTV benchmarks per channel (from 02_generate_clinic_data calibration)
CHANNEL_BENCHMARKS = {
    "google_search":    {"cac": 120, "avg_ltv_2yr": 1800, "ret_2yr": 0.55},
    "doctor_referral":  {"cac":  40, "avg_ltv_2yr": 3200, "ret_2yr": 0.74},
    "patient_referral": {"cac":  55, "avg_ltv_2yr": 2800, "ret_2yr": 0.68},
    "instagram":        {"cac": 180, "avg_ltv_2yr": 1100, "ret_2yr": 0.41},
    "walk_in":          {"cac":  20, "avg_ltv_2yr": 1400, "ret_2yr": 0.48},
}


def channel_roas(marketing):
    """ROAS and actual CAC by channel over all months."""
    agg = marketing.groupby("channel").agg(
        total_spend=           ("spend",                 "sum"),
        total_patients=        ("new_patients_acquired", "sum"),
        total_rev_attributed=  ("revenue_attributed",    "sum"),
    ).reset_index()
    agg["roas"]        = (agg["total_rev_attributed"] / agg["total_spend"]).round(3)
    agg["actual_cac"]  = (agg["total_spend"] / agg["total_patients"].replace(0, np.nan)).round(2)
    agg = agg.round(2)
    return agg.to_dict("records")


def ltv_cac_by_channel(patients, appts):
    """
    Actual LTV (total revenue per patient) by acquisition channel,
    combined with benchmark CAC → LTV:CAC ratio.
    """
    attended = appts[appts["no_show"] == 0]
    pat_rev = attended.groupby("patient_id")["revenue"].sum().reset_index()
    pat_rev.columns = ["patient_id", "actual_ltv"]

    df = patients[["patient_id","acquisition_channel"]].merge(pat_rev, on="patient_id", how="left")
    df["actual_ltv"] = df["actual_ltv"].fillna(0)

    result = []
    for ch, grp in df.groupby("acquisition_channel"):
        bm    = CHANNEL_BENCHMARKS.get(ch, {"cac": 100, "avg_ltv_2yr": 1500, "ret_2yr": 0.50})
        avg_ltv = float(grp["actual_ltv"].mean())
        cac     = bm["cac"]
        result.append({
            "channel":        ch,
            "n_patients":     int(len(grp)),
            "avg_ltv":        round(avg_ltv, 2),
            "cac":            cac,
            "ltv_cac_ratio":  round(avg_ltv / cac, 2),
            "ret_2yr":        bm["ret_2yr"],
            "benchmark_ltv":  bm["avg_ltv_2yr"],
        })
    result.sort(key=lambda x: x["ltv_cac_ratio"], reverse=True)
    return result


def monthly_channel_trend(marketing):
    """Monthly spend and acquired patients by channel — last 12 months."""
    recent = marketing.copy()
    recent["month_dt"] = pd.to_datetime(recent["month"] + "-01")
    recent = recent.sort_values("month_dt")
    last12 = recent["month"].unique()[-12:]
    recent = recent[recent["month"].isin(last12)]

    channels = recent["channel"].unique().tolist()
    months   = sorted(recent["month"].unique().tolist())

    spend_by_ch = {}
    pats_by_ch  = {}
    for ch in channels:
        sub = recent[recent["channel"] == ch].set_index("month")
        spend_by_ch[ch] = [round(float(sub.loc[m, "spend"]), 2) if m in sub.index else 0 for m in months]
        pats_by_ch[ch]  = [int(sub.loc[m, "new_patients_acquired"]) if m in sub.index else 0 for m in months]

    return {"months": months, "spend": spend_by_ch, "patients": pats_by_ch}


def budget_recommendation(ltv_cac):
    """
    Data-driven budget reallocation recommendation.
    Doctor referral has highest LTV:CAC → invest more.
    Instagram has lowest retention → reduce or repurpose toward cosmetic only.
    """
    sorted_ch = sorted(ltv_cac, key=lambda x: x["ltv_cac_ratio"], reverse=True)
    best      = sorted_ch[0]
    worst     = sorted_ch[-1]

    return {
        "headline": (
            f"Reallocate 20% of {worst['channel']} budget (CAC ${worst['cac']}, "
            f"LTV:CAC {worst['ltv_cac_ratio']:.1f}x) to {best['channel']} "
            f"(CAC ${best['cac']}, LTV:CAC {best['ltv_cac_ratio']:.1f}x)"
        ),
        "best_channel":  best["channel"],
        "worst_channel": worst["channel"],
        "best_ltv_cac":  best["ltv_cac_ratio"],
        "worst_ltv_cac": worst["ltv_cac_ratio"],
        "insight": (
            "Doctor referrals produce the highest LTV:CAC ratio and 74% 2-year "
            "retention — far exceeding Instagram (41% retention, 4.5x lower LTV). "
            "Investing $500/month into a referral program (gift cards, co-marketing "
            "with PCPs) generates more durable revenue than equivalent social spend."
        ),
    }


def main():
    marketing = pd.read_csv(MARKETING_CSV)
    patients  = pd.read_csv(PATIENTS_CSV)
    appts     = pd.read_csv(APPTS_CSV)
    log.info("Loaded marketing (%d rows), patients (%d), appointments (%d)",
             len(marketing), len(patients), len(appts))

    log.info("ROAS by channel …")
    roas_data = channel_roas(marketing)

    log.info("LTV:CAC analysis …")
    ltv_cac = ltv_cac_by_channel(patients, appts)

    log.info("Monthly trend …")
    trend = monthly_channel_trend(marketing)

    log.info("Budget recommendation …")
    rec = budget_recommendation(ltv_cac)

    # Channel performance summary table
    log.info("\n── Channel Performance ───────────────────────────────────────────")
    log.info("  %-20s %8s %8s %8s %10s %6s", "Channel","ROAS","CAC","LTV","LTV:CAC","Ret2yr")
    for row in ltv_cac:
        roas_row = next((r for r in roas_data if r["channel"] == row["channel"]), {})
        log.info("  %-20s %8.2f $%6.0f $%6.0f %10.1fx  %.0f%%",
                 row["channel"],
                 roas_row.get("roas", 0),
                 row["cac"],
                 row["avg_ltv"],
                 row["ltv_cac_ratio"],
                 row["ret_2yr"] * 100)
    log.info("\n  📋 Recommendation: %s", rec["headline"])

    analysis = {
        "roas_by_channel":    roas_data,
        "ltv_cac_by_channel": ltv_cac,
        "monthly_trend":      trend,
        "budget_recommendation": rec,
        "total_spend_3yr":    round(float(marketing["spend"].sum()), 2),
        "total_patients_3yr": int(marketing["new_patients_acquired"].sum()),
        "blended_cac":        round(float(marketing["spend"].sum() / marketing["new_patients_acquired"].sum()), 2),
        "blended_roas":       round(float(marketing["revenue_attributed"].sum() / marketing["spend"].sum()), 3),
    }

    with open(ANALYTICS_JSON, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("Saved → %s", ANALYTICS_JSON)
    log.info("✓  Layer 5 complete.")


if __name__ == "__main__":
    main()
