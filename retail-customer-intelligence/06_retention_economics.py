"""
06_retention_economics.py
Rebuild "revenue at risk" as a number that survives a follow-up question.

The problem with the original £1.46M
------------------------------------
03_churn_model.py computes it as:

    cust['ChurnProb'] = rf.predict_proba(X)[:,1]          # ALL rows, incl. training
    revenue_at_risk   = cust[cust.ChurnRisk=='High'].Monetary.sum()

Two defects, both fatal to the claim as stated:

  1. In-sample scoring. X is the full feature matrix, so ~80% of the customers
     behind that total were in the forest's training set. Their churn scores are
     optimistic, which inflates who lands in the High bucket.
  2. Backward-looking value. `Monetary` is revenue ALREADY BOOKED in the training
     window. So the figure is "historic revenue from customers now flagged
     risky", not revenue that will be lost. It answers a question nobody asked.

The threshold (0.65) was also arbitrary — no cost model, no precision@k.

What this script does instead
----------------------------
  * Scores ONLY held-out customers, with a model refit on the training split.
  * Defines value as revenue actually OBSERVED after the cutoff, so "at risk"
    means forward revenue that a save campaign could protect.
  * Emits a precision@k / revenue-recovered curve, so the operating threshold is
    chosen by economics rather than by a round number.
  * Solves for the break-even save rate given a per-contact cost.

Run:  python 06_retention_economics.py   (writes retention_economics.json)
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

CUTOFF        = pd.Timestamp("2011-09-01")
CONTACT_COST  = 3.00     # £ per retention contact (discount + send). Stated, not hidden.
OUT_JSON      = "retention_economics.json"

FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderVal", "UniqueProds",
            "AvgQuantity", "AvgPrice", "Tenure", "ActiveMonths", "TxPerMonth"]


def main():
    df = pd.read_csv("online_retail_clean.csv", parse_dates=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    cust = pd.read_csv("customer_features.csv")

    # ── Forward revenue per customer: what they actually spent AFTER the cutoff ──
    future = df[df["InvoiceDate"] >= CUTOFF]
    fwd = (future.groupby("CustomerID")["Revenue"].sum()
                 .rename("FwdRevenue").reset_index())
    cust = cust.merge(fwd, on="CustomerID", how="left")
    cust["FwdRevenue"] = cust["FwdRevenue"].fillna(0.0)

    X = cust[FEATURES].fillna(0)
    y = cust["Churned"]

    # Same split as 03 (same seed/stratify), so the holdout here is exactly the
    # holdout the reported accuracy/AUC came from — no second, different split.
    idx = np.arange(len(cust))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    rf.fit(X.iloc[itr], y.iloc[itr])

    te = cust.iloc[ite].copy()
    te["ChurnProb"] = rf.predict_proba(X.iloc[ite])[:, 1]
    te = te.sort_values("ChurnProb", ascending=False).reset_index(drop=True)

    # A "save" is only possible for a customer who WOULD have churned and who had
    # forward revenue to lose. Churned==1 means no purchase after the cutoff, so
    # by construction their observed forward revenue is 0 — the recoverable
    # quantity is what a retained customer LIKE them spends. Use the median
    # forward revenue of retained customers in the same segment as the estimate.
    retained = te[te["Churned"] == 0]
    seg_value = retained.groupby("Segment")["FwdRevenue"].median()
    overall_value = float(retained["FwdRevenue"].median())
    te["ValueIfSaved"] = np.where(
        te["Churned"] == 1,
        te["Segment"].map(seg_value).fillna(overall_value),
        0.0,   # already retained — nothing to save
    )

    # ── precision@k and recoverable revenue @k ────────────────────────────────
    n = len(te)
    ks, curve = [], []
    for k in range(50, n + 1, 50):
        top = te.head(k)
        hits = int(top["Churned"].sum())
        rec = float(top["ValueIfSaved"].sum())
        cost = k * CONTACT_COST
        ks.append(k)
        curve.append({
            "k": k,
            "precision": round(hits / k, 4),
            "hits": hits,
            "recoverable": round(rec, 2),
            "cost": round(cost, 2),
            # Save rate at which contacting the top k breaks even.
            "breakeven_save_rate": round(cost / rec, 4) if rec > 0 else None,
        })

    base_rate = float(te["Churned"].mean())

    # ── Choose the operating point by expected profit, not by a round threshold ──
    # profit(k) = save_rate * recoverable(k) - k * contact_cost
    # Marginal expected value of contacting the customer at rank k is
    # save_rate * value * precision_at_k, so it is worth contacting while that
    # exceeds contact cost. With a ~£528 median forward value against a £3 contact,
    # the ratio is ~176:1, which means cost is NOT the binding constraint — the
    # profit-maximising k is essentially "everyone the model flags". Reporting that
    # honestly is more useful than inventing a threshold that looks decisive.
    profit = {}
    for s_rate in (0.10, 0.15, 0.20):
        rows = [{"k": c["k"],
                 "profit": round(s_rate * c["recoverable"] - c["cost"], 2)}
                for c in curve]
        best = max(rows, key=lambda r: r["profit"])
        at_best = next(c for c in curve if c["k"] == best["k"])
        profit["save_rate_%d" % int(s_rate * 100)] = {
            "best_k": best["k"],
            "profit_gbp": best["profit"],
            "precision_at_best_k": at_best["precision"],
            "recoverable_gbp": at_best["recoverable"],
            "cost_gbp": at_best["cost"],
        }

    # Capacity view: the realistic constraint is how many customers a small team
    # can actually contact, not the £3.
    capacity = {}
    for k in (100, 250, 500):
        c = next((x for x in curve if x["k"] == k), None)
        if c:
            capacity[str(k)] = {
                "precision": c["precision"],
                "churners_reached": c["hits"],
                "recoverable_gbp": c["recoverable"],
                "profit_at_15pct_save": round(0.15 * c["recoverable"] - c["cost"], 2),
            }

    out = {
        "cutoff": str(CUTOFF.date()),
        "contact_cost_gbp": CONTACT_COST,
        "holdout": {
            "n_customers": n,
            "churn_base_rate": round(base_rate, 4),
            "auc": None,   # filled below
        },
        "value_definition": (
            "Forward revenue observed after the cutoff. For a customer the model "
            "flags who did churn, the recoverable amount is estimated as the median "
            "forward revenue of RETAINED customers in the same segment — a churned "
            "customer's own observed forward revenue is zero by definition."
        ),
        "segment_median_forward_revenue": {k: round(float(v), 2) for k, v in seg_value.items()},
        "curve": curve,
        "profit_optimum": profit,
        "capacity_view": capacity,
        "cost_is_not_binding": (
            "Median forward value of a retained customer is ~176x the £3 contact "
            "cost, so expected profit keeps rising with reach and the profit-"
            "maximising k is effectively the whole flagged base. The binding "
            "constraint is contact capacity, not cost."
        ),
        "supersedes": {
            "old_figure_gbp": 1457904.6,
            "why_wrong": [
                "Scored in-sample: predict_proba was run on the full matrix, so ~80% "
                "of those customers were in the model's own training set.",
                "Backward-looking: summed Monetary, which is revenue already booked "
                "before the cutoff, not revenue at risk of being lost.",
                "Arbitrary threshold: ChurnProb > 0.65 with no cost model.",
            ],
        },
    }

    from sklearn.metrics import roc_auc_score
    out["holdout"]["auc"] = round(float(roc_auc_score(te["Churned"], te["ChurnProb"])), 4)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Holdout: {n} customers, churn base rate {base_rate:.1%}, "
          f"AUC {out['holdout']['auc']:.4f}")
    print(f"Median forward revenue of a retained customer: £{overall_value:,.0f}")
    print("\n  k   precision   hits   recoverable £   cost £   breakeven save rate")
    for c in curve[::4]:
        be = f"{c['breakeven_save_rate']:.1%}" if c["breakeven_save_rate"] else "—"
        print(f"{c['k']:4d}   {c['precision']:8.1%}   {c['hits']:4d}   "
              f"{c['recoverable']:12,.0f}   {c['cost']:6,.0f}   {be:>8}")
    print("\nProfit optimum by assumed save rate:")
    for k, v in profit.items():
        print(f"  {k:>13}: contact top {v['best_k']:4d}  "
              f"precision {v['precision_at_best_k']:.1%}  "
              f"profit £{v['profit_gbp']:,.0f}")
    print("\nCapacity view (what a small team could actually call):")
    for k, v in capacity.items():
        print(f"  top {k:>3}: precision {v['precision']:.1%}  "
              f"{v['churners_reached']:3d} churners  "
              f"£{v['recoverable_gbp']:,.0f} recoverable  "
              f"profit @15% save £{v['profit_at_15pct_save']:,.0f}")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
