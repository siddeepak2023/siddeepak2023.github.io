"""
07_export_dashboard.py
Compile all pipeline outputs into data/dashboard_data.json
for the self-contained HTML dashboard.

Run:  python3 07_export_dashboard.py
"""

import json
import logging
import sqlite3
from datetime import datetime, date

import numpy as np
import pandas as pd

PATIENTS_CSV    = "data/patients.csv"
APPTS_CSV       = "data/appointments.csv"
MARKETING_CSV   = "data/marketing.csv"
ANALYTICS_JSON  = "data/patient_analytics.json"
COHORT_JSON     = "data/cohort_retention.json"
NOSHOW_JSON     = "data/noshow_metrics.json"
MARKETING_AJSON = "data/marketing_analysis.json"
FORECAST_JSON   = "data/revenue_forecast.json"
BENCHMARKS_JSON = "data/cms_benchmarks.json"
SEGMENTS_CSV    = "data/patient_segments.csv"
OUTPUT_JSON     = "data/dashboard_data.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COSMETIC_CODES = {"COSM-BOT","COSM-FIL","COSM-LAS","COSM-PEL","COSM-MCN","COSM-CSC","COSM-IPL"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_kpis(patients, appts, marketing):
    attended = appts[appts["no_show"] == 0]
    monthly_rev = (attended.groupby(
        pd.to_datetime(attended["appt_date"]).dt.to_period("M"))["revenue"]
        .sum())

    # Last 30 days patients
    last_date = pd.to_datetime(attended["appt_date"]).max()
    recent_30d = attended[pd.to_datetime(attended["appt_date"]) >= last_date - pd.Timedelta(days=30)]

    # Active patients (visited in last 12 months)
    cutoff_12m = last_date - pd.Timedelta(days=365)
    active_pts = attended[pd.to_datetime(attended["appt_date"]) >= cutoff_12m]["patient_id"].nunique()

    # Average LTV
    pat_rev = attended.groupby("patient_id")["revenue"].sum()
    avg_ltv = float(pat_rev.mean())

    # Best ROAS channel from marketing
    mktg_agg = marketing.groupby("channel").agg(
        spend=("spend","sum"), rev=("revenue_attributed","sum")).reset_index()
    mktg_agg["roas"] = mktg_agg["rev"] / mktg_agg["spend"]
    best_ch = mktg_agg.loc[mktg_agg["roas"].idxmax()]

    return {
        "active_patients":   int(active_pts),
        "total_patients":    int(patients["patient_id"].nunique()),
        "monthly_revenue":   round(float(monthly_rev.iloc[4:-4].mean()) if len(monthly_rev) > 8 else float(monthly_rev.mean()), 2),
        "avg_monthly_revenue": round(float(monthly_rev.mean()), 2),
        "noshow_rate":       round(float(appts["no_show"].mean() * 100), 1),
        "avg_ltv":           round(avg_ltv, 2),
        "best_roas_channel": str(best_ch["channel"]),
        "best_roas_value":   round(float(best_ch["roas"]), 2),
        "total_revenue_3yr": round(float(attended["revenue"].sum()), 2),
        "appointments_ytd":  int((pd.to_datetime(attended["appt_date"]).dt.year == 2024).sum()),
    }


def build_revenue_trend(appts):
    """Monthly stacked medical + cosmetic revenue for chart."""
    df = appts[appts["no_show"] == 0].copy()
    df["appt_date"] = pd.to_datetime(df["appt_date"])
    df["month"] = df["appt_date"].dt.to_period("M").astype(str)
    df["is_cosmetic"] = df["hcpcs_code"].isin(COSMETIC_CODES)

    pivot = df.groupby(["month","is_cosmetic"])["revenue"].sum().unstack(fill_value=0)
    pivot.columns = ["medical" if not c else "cosmetic" for c in pivot.columns]
    pivot = pivot.reset_index().sort_values("month")

    return [{"month": r["month"],
             "medical":  round(float(r.get("medical",0)),  2),
             "cosmetic": round(float(r.get("cosmetic",0)), 2),
             "total":    round(float(r.get("medical",0) + r.get("cosmetic",0)), 2)}
            for _, r in pivot.iterrows()]


def build_todays_at_risk(appts, noshow_metrics):
    """Simulate today's highest-risk appointment list."""
    high_risk = noshow_metrics.get("high_risk_appointments", [])[:10]
    if not high_risk:
        # Fall back to recent high-booking-advance appointments
        df = appts[appts["no_show"] == 0].tail(50).copy()
        df["noshow_prob"] = np.random.uniform(0.35, 0.75, len(df))
        high_risk = df.nlargest(10, "noshow_prob")[
            ["appt_id","patient_id","appt_date","appointment_type","insurance_type","reminder_sent"]
        ].to_dict("records")
    return high_risk


def build_segment_profiles(segments_csv, analytics):
    df = pd.read_csv(segments_csv)
    seg_stats = analytics.get("segment_summary", {})
    ltv_chan  = analytics.get("ltv_by_channel", {})

    profiles = []
    seg_colors = {
        "VIP Cosmetic":  "#F4A261",
        "Loyal Medical": "#0D7377",
        "At-Risk":       "#E76F51",
        "Lapsed":        "#6C757D",
    }
    seg_icons = {
        "VIP Cosmetic":  "💎",
        "Loyal Medical": "🏥",
        "At-Risk":       "⚠️",
        "Lapsed":        "⏸️",
    }
    for seg, stats in seg_stats.items():
        # Insurance mix
        seg_df = df[df["segment"] == seg]
        ins_mix = seg_df["insurance_type"].value_counts(normalize=True).round(3).to_dict() if len(seg_df) > 0 else {}
        profiles.append({
            "name":        seg,
            "color":       seg_colors.get(seg, "#999"),
            "icon":        seg_icons.get(seg, "👤"),
            "n":           int(stats["n"]),
            "avg_ltv":     round(float(stats["avg_ltv"]), 2),
            "avg_visits":  round(float(stats["avg_frequency"]), 1),
            "avg_recency": round(float(stats["avg_recency"]), 0),
            "insurance_mix": {k: round(v*100, 1) for k, v in ins_mix.items()},
        })
    profiles.sort(key=lambda x: x["avg_ltv"], reverse=True)
    return profiles


def build_cms_comparison(benchmarks, appts):
    """Side-by-side: our clinic vs national CMS averages."""
    attended = appts[appts["no_show"] == 0]
    med_appts = attended[~attended["hcpcs_code"].isin(COSMETIC_CODES)]

    our_avg_payment = float(med_appts["revenue"].mean()) if len(med_appts) > 0 else 0
    nat_avg_payment = benchmarks.get("national_avg_payment_per_service", 106)
    tx_avg_payment  = benchmarks.get("tx_benchmarks", {}).get("avg_payment", nat_avg_payment)

    # Procedure mix comparison
    our_mix = med_appts.groupby("hcpcs_code")["appt_id"].count()
    our_mix_pct = (our_mix / our_mix.sum() * 100).round(1).to_dict()

    top_proc_cms = {r["HCPCS_Cd"]: r for r in benchmarks.get("top20_by_volume", [])}

    return {
        "our_avg_payment":  round(our_avg_payment, 2),
        "national_avg":     round(nat_avg_payment, 2),
        "tx_avg":           round(tx_avg_payment, 2),
        "clinic_vs_nat_pct":round((our_avg_payment / nat_avg_payment - 1) * 100, 1) if nat_avg_payment else 0,
        "our_procedure_mix": our_mix_pct,
        "cosmetic_gap_note": benchmarks.get("cosmetic_gap", {}).get("note", ""),
        "top20_national":    [
            {"code": r["HCPCS_Cd"], "desc": r["HCPCS_Desc"],
             "avg_payment": round(r["avg_payment"], 2),
             "nat_volume":  r["nat_total_srvcs"]}
            for r in benchmarks.get("top20_by_volume", [])[:10]
        ],
    }


def main():
    log.info("Loading all data …")
    patients   = pd.read_csv(PATIENTS_CSV)
    appts      = pd.read_csv(APPTS_CSV)
    marketing  = pd.read_csv(MARKETING_CSV)
    analytics  = load_json(ANALYTICS_JSON)
    cohort     = load_json(COHORT_JSON)
    noshow     = load_json(NOSHOW_JSON)
    mktg_anal  = load_json(MARKETING_AJSON)
    forecast   = load_json(FORECAST_JSON)
    benchmarks = load_json(BENCHMARKS_JSON)

    log.info("Building KPIs …")
    kpis = build_kpis(patients, appts, marketing)

    log.info("Building revenue trend …")
    rev_trend = build_revenue_trend(appts)

    log.info("Building segment profiles …")
    segments = build_segment_profiles(SEGMENTS_CSV, analytics)

    log.info("Building CMS comparison …")
    cms_comp = build_cms_comparison(benchmarks, appts)

    log.info("Building today's at-risk list …")
    at_risk = build_todays_at_risk(appts, noshow)

    dashboard_data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "clinic_name":  "DermCare Texas — Analytics Demo",
        "date":         date.today().isoformat(),
        "kpis":         kpis,
        "revenue_trend":rev_trend,
        "segments":     segments,
        "cohort_retention": cohort,
        "noshow": {
            "auc":              noshow["auc"],
            "accuracy":         noshow["accuracy"],
            "noshow_rate":      noshow["noshow_rate"],
            "confusion_matrix": noshow["confusion_matrix"],
            "roc_curve":        noshow["roc_curve"],
            "feature_importance": noshow["feature_importance"][:10],
            "business_impact":  noshow["business_impact"],
            "high_risk_today":  at_risk,
        },
        "marketing":    mktg_anal,
        "revenue_forecast": forecast,
        "cms_benchmarks": cms_comp,
        "raw_benchmarks": {
            "source":            benchmarks.get("source",""),
            "n_providers":       benchmarks.get("n_providers", 0),
            "n_states":          benchmarks.get("n_states", 0),
            "mohs_avg_payment":  benchmarks.get("mohs_avg_payment", 790),
            "geographic_variation": benchmarks.get("geographic_variation", {}),
            "procedure_mix_pct": benchmarks.get("procedure_mix_pct", {}),
            "high_value_procedures": benchmarks.get("high_value_procedures", []),
        },
        "patient_analytics": {
            "avg_ltv":           analytics["avg_ltv"],
            "median_ltv":        analytics["median_ltv"],
            "revenue_concentration": analytics["revenue_concentration"],
            "churn_model":       analytics["churn_model"],
            "ltv_by_channel":    analytics["ltv_by_channel"],
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    log.info("Saved → %s", OUTPUT_JSON)

    log.info("\n── Dashboard Export Summary ──────────────────────────────────────")
    log.info("  Active patients:   %d", kpis["active_patients"])
    log.info("  Monthly revenue:   $%.0f", kpis["monthly_revenue"])
    log.info("  No-show rate:      %.1f%%", kpis["noshow_rate"])
    log.info("  Avg LTV:           $%.0f", kpis["avg_ltv"])
    log.info("  Best ROAS channel: %s (%.2fx)", kpis["best_roas_channel"], kpis["best_roas_value"])
    log.info("✓  Layer 7 complete. Ready to build dashboard.")


if __name__ == "__main__":
    main()
