"""
03_patient_analytics.py
RFM segmentation, LTV analysis, cohort retention, churn prediction.
Outputs:
  data/patient_segments.csv
  data/cohort_retention.json
  data/patient_analytics.json

Run:  python3 03_patient_analytics.py
"""

import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

PATIENTS_CSV   = "data/patients.csv"
APPTS_CSV      = "data/appointments.csv"
SEGMENTS_CSV   = "data/patient_segments.csv"
ANALYTICS_JSON = "data/patient_analytics.json"
COHORT_JSON    = "data/cohort_retention.json"
SNAPSHOT_DATE  = datetime(2024, 12, 31)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def compute_rfm(patients, appts):
    """Recency (days since last visit), Frequency (# visits), Monetary (total revenue)."""
    snap = SNAPSHOT_DATE
    attended = appts[appts["no_show"] == 0].copy()
    attended["appt_date"] = pd.to_datetime(attended["appt_date"])

    last_visit = attended.groupby("patient_id")["appt_date"].max()
    frequency  = attended.groupby("patient_id")["appt_id"].count()
    monetary   = attended.groupby("patient_id")["revenue"].sum()

    rfm = patients[["patient_id","acquisition_channel","primary_condition",
                    "insurance_type","age","gender"]].copy()
    rfm = rfm.set_index("patient_id")
    rfm["recency"]   = (snap - last_visit).dt.days.fillna(snap.toordinal())
    rfm["frequency"] = frequency.reindex(rfm.index).fillna(0)
    rfm["monetary"]  = monetary.reindex(rfm.index).fillna(0)
    return rfm.reset_index()


def kmeans_segment(rfm, k=4, seed=42):
    """K-Means on scaled RFM → 4 named segments."""
    scaler   = StandardScaler()
    features = ["recency", "frequency", "monetary"]
    X_scaled = scaler.fit_transform(rfm[features].fillna(0))

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    rfm = rfm.copy()
    rfm["cluster"] = km.fit_predict(X_scaled)

    # Characterise clusters → assign business names
    cluster_stats = rfm.groupby("cluster")[features].mean()
    # Best cluster = low recency, high freq, high monetary → "VIP Cosmetic"
    # Sort by monetary desc to assign labels
    cluster_stats["score"] = (
        -cluster_stats["recency"] / cluster_stats["recency"].max()
        + cluster_stats["frequency"] / cluster_stats["frequency"].max()
        + cluster_stats["monetary"] / cluster_stats["monetary"].max()
    )
    ranked = cluster_stats["score"].sort_values(ascending=False).index.tolist()
    labels = ["VIP Cosmetic", "Loyal Medical", "At-Risk", "Lapsed"]
    label_map = {ranked[i]: labels[i] for i in range(k)}
    rfm["segment"] = rfm["cluster"].map(label_map)
    return rfm


def cohort_retention(patients, appts):
    """
    Cohort by acquisition quarter. For each cohort, track % still active
    at 6mo, 12mo, 18mo, 24mo.
    """
    patients = patients.copy()
    patients["first_visit_date"] = pd.to_datetime(patients["first_visit_date"])
    patients["cohort"] = patients["first_visit_date"].dt.to_period("Q").astype(str)

    appts = appts.copy()
    appts["appt_date"] = pd.to_datetime(appts["appt_date"])

    checkpoints = [6, 12, 18, 24]  # months
    result = []

    for cohort, grp in patients.groupby("cohort"):
        cohort_start = grp["first_visit_date"].min()
        n_cohort = len(grp)
        pids = set(grp["patient_id"])
        row = {"cohort": cohort, "n_patients": n_cohort}
        for m in checkpoints:
            window_start = cohort_start + pd.DateOffset(months=m-1)
            window_end   = cohort_start + pd.DateOffset(months=m+1)
            active = appts[
                (appts["patient_id"].isin(pids)) &
                (appts["appt_date"] >= window_start) &
                (appts["appt_date"] <= window_end) &
                (appts["no_show"] == 0)
            ]["patient_id"].nunique()
            row[f"ret_{m}mo"] = round(active / n_cohort, 4) if n_cohort > 0 else 0
        result.append(row)

    return pd.DataFrame(result)


def ltv_by_segment_and_channel(rfm):
    """LTV (monetary) broken down by segment and acquisition channel."""
    by_seg = rfm.groupby("segment")["monetary"].agg(
        avg_ltv="mean", median_ltv="median", n="count").round(2).to_dict("index")
    by_chan = rfm.groupby("acquisition_channel")["monetary"].agg(
        avg_ltv="mean", median_ltv="median", n="count").round(2).to_dict("index")
    return by_seg, by_chan


def revenue_concentration(rfm):
    """What % of revenue comes from top 10% / 20% of patients."""
    sorted_m = rfm["monetary"].sort_values(ascending=False)
    total = sorted_m.sum()
    n = len(sorted_m)
    top10_rev = sorted_m.head(max(1, n // 10)).sum()
    top20_rev = sorted_m.head(max(1, n // 5)).sum()
    top1_rev  = sorted_m.head(max(1, n // 100)).sum()
    return {
        "top_1pct_revenue_share":  round(float(top1_rev  / total), 4),
        "top_10pct_revenue_share": round(float(top10_rev / total), 4),
        "top_20pct_revenue_share": round(float(top20_rev / total), 4),
        "gini_coefficient":        round(float(_gini(sorted_m.values)), 4),
    }


def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def churn_model(rfm, patients):
    """
    Churn = no visit in last 18 months from snapshot.
    Logistic regression on RFM + demographic features.
    """
    df = rfm.copy()
    df["churned"] = (df["recency"] > 548).astype(int)  # 18 months ≈ 548 days

    # Exclude recency — it directly defines churn (would cause data leakage)
    feature_cols = ["frequency", "monetary", "age"]
    # Encode categoricals
    df["insurance_enc"] = df["insurance_type"].map(
        {"private": 3, "medicare": 2, "self_pay": 1, "medicaid": 0}).fillna(1)
    df["channel_enc"] = df["acquisition_channel"].map(
        {"doctor_referral": 4, "patient_referral": 3, "google_search": 2,
         "walk_in": 1, "instagram": 0}).fillna(2)
    feature_cols += ["insurance_enc", "channel_enc"]

    X = df[feature_cols].fillna(0).values
    y = df["churned"].values

    # Time-based split: first 80% by recency = train
    df_sorted = df.sort_values("recency")
    split_idx = int(len(df_sorted) * 0.80)
    train_idx = df_sorted.index[:split_idx]
    test_idx  = df_sorted.index[split_idx:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df.loc[train_idx, feature_cols].fillna(0))
    X_te = scaler.transform(df.loc[test_idx, feature_cols].fillna(0))
    y_tr = df.loc[train_idx, "churned"].values
    y_te = df.loc[test_idx,  "churned"].values

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)
    prob_te = model.predict_proba(X_te)[:, 1]
    auc = round(float(roc_auc_score(y_te, prob_te)), 4)
    acc = round(float((model.predict(X_te) == y_te).mean()), 4)

    log.info("  Churn model: AUC=%.4f  acc=%.4f  churn_rate=%.1f%%",
             auc, acc, y.mean() * 100)

    coefs = dict(zip(feature_cols, model.coef_[0].tolist()))
    return {
        "auc": auc, "accuracy": acc,
        "churn_rate": round(float(y.mean()), 4),
        "n_churned": int(y.sum()),
        "n_active":  int((y == 0).sum()),
        "feature_coefs": {k: round(v, 4) for k, v in coefs.items()},
    }


def main():
    patients = pd.read_csv(PATIENTS_CSV)
    appts    = pd.read_csv(APPTS_CSV)
    log.info("Loaded %d patients, %d appointments", len(patients), len(appts))

    log.info("Computing RFM …")
    rfm = compute_rfm(patients, appts)

    log.info("K-Means segmentation (k=4) …")
    rfm = kmeans_segment(rfm)

    log.info("Cohort retention …")
    cohort_df = cohort_retention(patients, appts)

    log.info("LTV by segment and channel …")
    ltv_seg, ltv_chan = ltv_by_segment_and_channel(rfm)

    log.info("Revenue concentration …")
    rev_conc = revenue_concentration(rfm)

    log.info("Churn model …")
    churn = churn_model(rfm, patients)

    # Segment summary
    seg_summary = rfm.groupby("segment").agg(
        n=("patient_id","count"),
        avg_recency=("recency","mean"),
        avg_frequency=("frequency","mean"),
        avg_ltv=("monetary","mean"),
    ).round(2).to_dict("index")

    log.info("\n  Segments:")
    for seg, stats in seg_summary.items():
        log.info("    %-18s  n=%4d  avg_ltv=$%.0f  avg_visits=%.1f",
                 seg, stats["n"], stats["avg_ltv"], stats["avg_frequency"])

    # Save outputs
    rfm.to_csv(SEGMENTS_CSV, index=False)
    log.info("Saved → %s", SEGMENTS_CSV)

    cohort_records = []
    for _, row in cohort_df.iterrows():
        cohort_records.append({
            "cohort":     row["cohort"],
            "n_patients": int(row["n_patients"]),
            "ret_6mo":    float(row.get("ret_6mo", 0)),
            "ret_12mo":   float(row.get("ret_12mo", 0)),
            "ret_18mo":   float(row.get("ret_18mo", 0)),
            "ret_24mo":   float(row.get("ret_24mo", 0)),
        })
    with open(COHORT_JSON, "w") as f:
        json.dump(cohort_records, f, indent=2)
    log.info("Saved → %s", COHORT_JSON)

    analytics = {
        "segment_summary": seg_summary,
        "ltv_by_segment":  ltv_seg,
        "ltv_by_channel":  ltv_chan,
        "revenue_concentration": rev_conc,
        "churn_model":     churn,
        "total_patients":  int(len(patients)),
        "avg_ltv":         round(float(rfm["monetary"].mean()), 2),
        "median_ltv":      round(float(rfm["monetary"].median()), 2),
    }
    with open(ANALYTICS_JSON, "w") as f:
        json.dump(analytics, f, indent=2)
    log.info("Saved → %s", ANALYTICS_JSON)

    log.info("\n── Patient Analytics Summary ─────────────────────────────────────")
    log.info("  Avg LTV:            $%.2f", analytics["avg_ltv"])
    log.info("  Top 10%% revenue:   %.1f%% of total", rev_conc["top_10pct_revenue_share"]*100)
    log.info("  Churn rate:         %.1f%%", churn["churn_rate"]*100)
    log.info("  Churn AUC:          %.4f", churn["auc"])
    log.info("✓  Layer 3 complete.")


if __name__ == "__main__":
    main()
