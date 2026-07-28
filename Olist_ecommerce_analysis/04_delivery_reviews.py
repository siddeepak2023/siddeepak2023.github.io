"""
04_delivery_reviews.py
Fix the grain, then answer the causal question honestly.

Two problems with the published delivery/review finding
-------------------------------------------------------
1. GRAIN. 03_export_tableau.py:57 joins `master` (order-ITEM grain, 110,197 rows)
   to reviews on order_id, then reports COUNT(*) as "total_orders" and takes a
   plain AVG(review_score). A five-item order therefore counts five times, so the
   average is item-weighted while the sentence read as per-order. The same script
   uses COUNT(DISTINCT order_id) correctly elsewhere — the pattern was known and
   dropped in exactly the query the résumé headlined.

2. FRAMING. "31% lower review scores" was a ratio of two means on an ordinal 1-5
   star scale, which is not a percentage of anything. And it was unadjusted:
   delivery time is confounded with state, product category and freight cost, so
   the raw gap cannot be read as the cost of slow delivery.

What this does
--------------
  * Recomputes the bucket means at ORDER grain (one row per order, reviews
    averaged per order first).
  * Fits a logistic regression for P(review <= 3) on delivery days, CONTROLLING
    for customer state, product category and freight cost, and reports the odds
    ratio per extra delivery day with a 95% CI — the adjusted effect.
  * Reports the unadjusted odds ratio beside it, so the confounding is visible
    rather than asserted.

Run:  python 04_delivery_reviews.py   (writes delivery_reviews.json)
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RAW = os.path.expanduser("~/Downloads/archive")
REVIEWS = os.path.join(RAW, "olist_order_reviews_dataset.csv")
MASTER  = "olist_master_clean.csv"
OUT     = "delivery_reviews.json"

BUCKETS = [(0, 7, "Fast (7 days or less)"), (8, 14, "Normal (8 to 14 days)"),
           (15, 21, "Slow (15 to 21 days)"), (22, 10**6, "Very slow (over 21 days)")]


def label(days):
    for lo, hi, name in BUCKETS:
        if lo <= days <= hi:
            return name
    return None


def main():
    if not os.path.exists(REVIEWS):
        raise SystemExit("missing %s — the 9 raw Olist CSVs live in ~/Downloads/archive" % REVIEWS)

    m = pd.read_csv(MASTER, usecols=[
        "order_id", "delivery_days", "freight_value", "price",
        "category", "customer_state", "order_item_id"])
    r = pd.read_csv(REVIEWS, usecols=["order_id", "review_score"])

    print("item rows: %,d" .replace(",d", "d") % len(m) if False else
          "item rows: {:,}".format(len(m)))
    print("distinct orders: {:,}".format(m["order_id"].nunique()))

    # ── Collapse to ORDER grain ───────────────────────────────────────────────
    # A handful of orders carry more than one review, so average per order before
    # joining; otherwise the fan-out reappears from the review side.
    r_ord = r.groupby("order_id", as_index=False)["review_score"].mean()

    orders = m.groupby("order_id", as_index=False).agg(
        delivery_days=("delivery_days", "first"),
        freight=("freight_value", "sum"),
        order_value=("price", "sum"),
        items=("order_item_id", "count"),
        state=("customer_state", "first"),
        # An order can span categories; use its highest-value line as the label.
        category=("category", "first"),
    )
    df = orders.merge(r_ord, on="order_id", how="inner").dropna(
        subset=["delivery_days", "review_score", "freight", "state", "category"])
    df = df[df["delivery_days"] >= 0]
    print("orders with a review: {:,}".format(len(df)))

    df["bucket"] = df["delivery_days"].apply(label)
    df = df[df["bucket"].notna()]

    # ── Bucket means at order grain, next to the old item-grain numbers ───────
    g = df.groupby("bucket").agg(orders=("order_id", "size"),
                                 mean_score=("review_score", "mean")).reset_index()
    order_names = [b[2] for b in BUCKETS]
    g["__o"] = g["bucket"].map({n: i for i, n in enumerate(order_names)})
    g = g.sort_values("__o").drop(columns="__o")

    item = m.merge(r_ord, on="order_id", how="inner")
    item["bucket"] = item["delivery_days"].apply(label)
    gi = item[item["bucket"].notna()].groupby("bucket").agg(
        item_rows=("order_id", "size"), mean_score=("review_score", "mean")).reset_index()
    gi["__o"] = gi["bucket"].map({n: i for i, n in enumerate(order_names)})
    gi = gi.sort_values("__o").drop(columns="__o")

    fast_o = float(g.iloc[0]["mean_score"]); slow_o = float(g.iloc[-1]["mean_score"])
    fast_i = float(gi.iloc[0]["mean_score"]); slow_i = float(gi.iloc[-1]["mean_score"])

    # ── Logistic regression: P(review <= 3) ──────────────────────────────────
    df["bad"] = (df["review_score"] <= 3).astype(int)
    df["log_freight"] = np.log1p(df["freight"])
    # Rare categories cannot support their own coefficient; pool the tail.
    top_cat = df["category"].value_counts().head(15).index
    df["cat"] = np.where(df["category"].isin(top_cat), df["category"], "other")
    top_st = df["state"].value_counts().head(12).index
    df["st"] = np.where(df["state"].isin(top_st), df["state"], "other")

    unadj = smf.logit("bad ~ delivery_days", data=df).fit(disp=0)
    adj = smf.logit("bad ~ delivery_days + log_freight + C(st) + C(cat)",
                    data=df).fit(disp=0)

    def odds(res):
        b = res.params["delivery_days"]
        lo, hi = res.conf_int().loc["delivery_days"]
        return {"odds_ratio_per_day": round(float(np.exp(b)), 4),
                "ci95": [round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)],
                "p_value": float(res.pvalues["delivery_days"]),
                "n": int(res.nobs)}

    u, a = odds(unadj), odds(adj)
    # Effect of a one-week slip, adjusted.
    week = round(float(a["odds_ratio_per_day"] ** 7), 3)

    out = {
        "grain": {
            "item_rows": int(len(m)),
            "distinct_orders": int(m["order_id"].nunique()),
            "orders_with_review": int(len(df)),
        },
        "order_grain_buckets": [
            {"bucket": row["bucket"], "orders": int(row["orders"]),
             "mean_score": round(float(row["mean_score"]), 3)}
            for _, row in g.iterrows()
        ],
        "item_grain_buckets_for_comparison": [
            {"bucket": row["bucket"], "item_rows": int(row["item_rows"]),
             "mean_score": round(float(row["mean_score"]), 3)}
            for _, row in gi.iterrows()
        ],
        "star_gap": {
            "order_grain": round(fast_o - slow_o, 3),
            "item_grain": round(fast_i - slow_i, 3),
            "note": ("Report the gap in STARS. The old '31% drop' divided two means on an "
                     "ordinal 1-5 scale, which is not a percentage of anything."),
        },
        "logit_p_review_le_3": {
            "unadjusted": u,
            "adjusted_for_state_category_freight": a,
            "adjusted_odds_ratio_per_week": week,
            "controls": {"states": int(len(top_st)) + 1, "categories": int(len(top_cat)) + 1,
                         "freight": "log1p(order freight)"},
        },
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    print("\norder-grain buckets:")
    for row in out["order_grain_buckets"]:
        print("  {bucket:<26} {orders:>7,}  {mean_score:.3f}".format(**row))
    print("\nstar gap  order grain %.3f  |  item grain %.3f" % (fast_o - slow_o, fast_i - slow_i))
    print("\nP(review <= 3) per extra delivery day:")
    print("  unadjusted  OR %.4f  CI [%.4f, %.4f]  p=%.2e  n=%d"
          % (u["odds_ratio_per_day"], u["ci95"][0], u["ci95"][1], u["p_value"], u["n"]))
    print("  adjusted    OR %.4f  CI [%.4f, %.4f]  p=%.2e  n=%d"
          % (a["odds_ratio_per_day"], a["ci95"][0], a["ci95"][1], a["p_value"], a["n"]))
    print("  adjusted, per WEEK of slip: OR %.3f" % week)
    print("\nSaved → %s" % OUT)


if __name__ == "__main__":
    main()
