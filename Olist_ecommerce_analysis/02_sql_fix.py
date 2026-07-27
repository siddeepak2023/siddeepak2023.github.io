import sqlite3
import pandas as pd

conn = sqlite3.connect("olist.db")

# Add reviews table if missing
reviews = pd.read_csv("olist_order_reviews_dataset.csv")
reviews.to_sql("reviews", conn, if_exists="replace", index=False)
print("Reviews table loaded.\n")

def run(label, sql):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print('─'*60)
    df = pd.read_sql_query(sql, conn)
    print(df.to_string(index=False))
    return df

# ── QUERY 6 FIX: Delivery speed vs review score ──────────────────────────────
q6 = run("Delivery speed vs customer review score", """
    WITH delivery_buckets AS (
        SELECT
            m.order_id,
            m.delivery_days,
            r.review_score,
            CASE
                WHEN m.delivery_days <= 7  THEN '1. Fast (7 days or less)'
                WHEN m.delivery_days <= 14 THEN '2. Normal (8 to 14 days)'
                WHEN m.delivery_days <= 21 THEN '3. Slow (15 to 21 days)'
                ELSE '4. Very slow (over 21 days)'
            END AS delivery_bucket
        FROM master m
        JOIN reviews r ON m.order_id = r.order_id
        WHERE m.delivery_days IS NOT NULL
          AND r.review_score IS NOT NULL
    )
    SELECT
        delivery_bucket,
        COUNT(*)                     AS total_orders,
        ROUND(AVG(review_score), 2)  AS avg_review_score,
        ROUND(AVG(delivery_days), 1) AS avg_delivery_days
    FROM delivery_buckets
    GROUP BY delivery_bucket
    ORDER BY delivery_bucket
""")

# ── QUERY 7: Top 10 sellers by revenue ───────────────────────────────────────
q7 = run("Top 10 sellers by revenue", """
    SELECT
        m.seller_id,
        s.seller_state,
        COUNT(DISTINCT m.order_id)  AS total_orders,
        ROUND(SUM(m.revenue), 2)    AS total_revenue,
        ROUND(AVG(m.revenue), 2)    AS avg_order_value
    FROM master m
    JOIN sellers s ON m.seller_id = s.seller_id
    GROUP BY m.seller_id, s.seller_state
    ORDER BY total_revenue DESC
    LIMIT 10
""")

# ── SAVE REMAINING RESULTS ────────────────────────────────────────────────────
print("\n\nSaving results...")
with pd.ExcelWriter("olist_sql_results_q6q7.xlsx") as writer:
    q6.to_excel(writer, sheet_name="Delivery vs Reviews", index=False)
    q7.to_excel(writer, sheet_name="Top Sellers",         index=False)

print("Saved: olist_sql_results_q6q7.xlsx")
print("All 7 queries complete.")

conn.close()
