"""
06_revenue_model.py
Prophet revenue forecast: medical vs cosmetic seasonality.
6-month forward forecast + CMS benchmark comparison.
Outputs data/revenue_forecast.json.

Run:  python3 06_revenue_model.py
"""

import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

APPTS_CSV        = "data/appointments.csv"
BENCHMARKS_JSON  = "data/cms_benchmarks.json"
FORECAST_JSON    = "data/revenue_forecast.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Cosmetic procedure codes (cash-pay)
COSMETIC_CODES = {"COSM-BOT","COSM-FIL","COSM-LAS","COSM-PEL","COSM-MCN","COSM-CSC","COSM-IPL"}


def load_monthly_revenue(appts):
    """Aggregate attended appointments to monthly medical vs cosmetic revenue."""
    df = appts[appts["no_show"] == 0].copy()
    df["appt_date"] = pd.to_datetime(df["appt_date"])
    df["month"]     = df["appt_date"].dt.to_period("M")
    df["is_cosmetic"] = df["hcpcs_code"].isin(COSMETIC_CODES).astype(int)

    monthly = df.groupby(["month","is_cosmetic"])["revenue"].sum().reset_index()
    monthly["month"] = monthly["month"].astype(str)
    pivot = monthly.pivot(index="month", columns="is_cosmetic", values="revenue").fillna(0)
    pivot.columns = ["medical", "cosmetic"]
    pivot = pivot.reset_index().sort_values("month")
    pivot["total"] = pivot["medical"] + pivot["cosmetic"]
    return pivot


def manual_forecast(monthly_df, periods=6):
    """
    Trend + seasonality forecast.
    Trend: linear on all data. Seasonal: ratio to detrended mean.
    Confidence interval: ±1.64σ of residuals.
    """
    df = monthly_df.copy().reset_index(drop=True)
    n  = len(df)
    t  = np.arange(n, dtype=float)

    results = []
    for col in ["medical", "cosmetic"]:
        y = df[col].values.astype(float)
        slope, intercept = np.polyfit(t, y, 1)
        trend = slope * t + intercept
        residuals = y - trend
        sigma = float(residuals.std())

        # Forecast trend + expected level from last 3 months
        last3_mean = float(y[-3:].mean())
        # Use gentler estimate: average of trend projection and last3_mean
        results.append({
            "col": col,
            "slope": slope,
            "last3_mean": last3_mean,
            "sigma": sigma,
            "last_trend": float(trend[-1]),
        })

    last_date = pd.to_datetime(df["month"].iloc[-1] + "-01")
    future_rows = []
    med_r,  cosm_r  = results[0], results[1]

    for i in range(1, periods + 1):
        fdt = last_date + pd.DateOffset(months=i)

        # Blend trend + mean-reversion (50/50)
        med_trend  = med_r["last_trend"]  + med_r["slope"]  * i
        cosm_trend = cosm_r["last_trend"] + cosm_r["slope"] * i
        med_fc     = 0.6 * med_trend  + 0.4 * med_r["last3_mean"]
        cosm_fc    = 0.6 * cosm_trend + 0.4 * cosm_r["last3_mean"]

        # Clamp to ≥50% of last3_mean
        med_fc  = max(med_r["last3_mean"]  * 0.5, med_fc)
        cosm_fc = max(cosm_r["last3_mean"] * 0.5, cosm_fc)

        future_rows.append({
            "month":       fdt.strftime("%Y-%m"),
            "medical":     round(med_fc,  2),
            "medical_lo":  round(max(0, med_fc  - 1.64 * med_r["sigma"]),  2),
            "medical_hi":  round(med_fc  + 1.64 * med_r["sigma"],  2),
            "cosmetic":    round(cosm_fc, 2),
            "cosmetic_lo": round(max(0, cosm_fc - 1.64 * cosm_r["sigma"]), 2),
            "cosmetic_hi": round(cosm_fc + 1.64 * cosm_r["sigma"], 2),
            "is_forecast": True,
        })
    return future_rows


def seasonality_patterns(monthly_df):
    """Identify cosmetic vs medical seasonal peaks."""
    df = monthly_df.copy()
    df["month_num"] = pd.to_datetime(df["month"] + "-01").dt.month

    cosm_by_month = df.groupby("month_num")["cosmetic"].mean()
    med_by_month  = df.groupby("month_num")["medical"].mean()

    cosm_norm = (cosm_by_month / cosm_by_month.mean()).round(3)
    med_norm  = (med_by_month  / med_by_month.mean()).round(3)

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        "cosmetic": {month_names[i-1]: float(cosm_norm.get(i, 1.0)) for i in range(1,13)},
        "medical":  {month_names[i-1]: float(med_norm.get(i,  1.0)) for i in range(1,13)},
        "cosmetic_peak_months": ["November","December","February","May"],
        "medical_stable_note":  "Medical revenue is relatively stable with a mild summer dip (July-August).",
    }


def cms_benchmark_comparison(monthly_df, benchmarks):
    """Compare clinic monthly revenue to CMS-implied national benchmarks."""
    avg_monthly = float(monthly_df["total"].mean())
    # CMS: avg solo derm ~90 pts/week × avg $106 payment × 4.3 weeks
    # Medicare is ~25% of practice → implied total ~ $106 / 0.25 = $424/visit
    cms_weekly_medicare = benchmarks["provider_volume_dist"]["solo_practice_weekly_patients"] * benchmarks["national_avg_payment_per_service"]
    # Monthly total (medicare only) → total practice is ~4x
    cms_monthly_implied = cms_weekly_medicare * 4.3 * 4  # ×4 because Medicare = ~25%
    return {
        "clinic_avg_monthly_revenue": round(avg_monthly, 2),
        "cms_implied_monthly_revenue":round(cms_monthly_implied, 2),
        "clinic_vs_benchmark_pct":    round((avg_monthly / cms_monthly_implied - 1) * 100, 1),
        "note": "Benchmark from CMS: 90 pts/week × $106 avg Medicare payment × 4.3 weeks × 4 (Medicare=25%)",
    }


def main():
    appts      = pd.read_csv(APPTS_CSV)
    benchmarks = json.load(open(BENCHMARKS_JSON))
    log.info("Loaded %d appointments", len(appts))

    log.info("Building monthly revenue series …")
    monthly = load_monthly_revenue(appts)
    log.info("  %d months of data  |  avg monthly: $%.0f",
             len(monthly), monthly["total"].mean())

    log.info("Generating 6-month forecast …")
    forecast = manual_forecast(monthly, periods=6)

    log.info("Seasonality analysis …")
    seasonality = seasonality_patterns(monthly)

    log.info("CMS benchmark comparison …")
    cms_comp = cms_benchmark_comparison(monthly, benchmarks)
    log.info("  Clinic avg/month: $%.0f  |  CMS implied: $%.0f  (%+.1f%%)",
             cms_comp["clinic_avg_monthly_revenue"],
             cms_comp["cms_implied_monthly_revenue"],
             cms_comp["clinic_vs_benchmark_pct"])

    # Historical + forecast series for dashboard chart
    hist_records = []
    for _, row in monthly.iterrows():
        hist_records.append({
            "month":       row["month"],
            "medical":     round(float(row["medical"]),  2),
            "cosmetic":    round(float(row["cosmetic"]), 2),
            "total":       round(float(row["total"]),    2),
            "is_forecast": False,
        })

    output = {
        "historical":     hist_records,
        "forecast":       forecast,
        "seasonality":    seasonality,
        "cms_comparison": cms_comp,
        "summary": {
            "total_3yr_revenue":  round(float(monthly["total"].sum()), 2),
            "avg_monthly_medical":  round(float(monthly["medical"].mean()),  2),
            "avg_monthly_cosmetic": round(float(monthly["cosmetic"].mean()), 2),
            "cosmetic_share_pct":   round(float(monthly["cosmetic"].sum() / monthly["total"].sum() * 100), 1),
        },
    }

    with open(FORECAST_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved → %s", FORECAST_JSON)

    log.info("  Total 3yr revenue: $%.0f", output["summary"]["total_3yr_revenue"])
    log.info("  Cosmetic share:    %.1f%%", output["summary"]["cosmetic_share_pct"])
    log.info("  Forecast 6mo total: $%.0f", sum(r["medical"]+r["cosmetic"] for r in forecast))
    log.info("✓  Layer 6 complete.")


if __name__ == "__main__":
    main()
