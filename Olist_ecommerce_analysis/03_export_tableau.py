import sqlite3
import pandas as pd

conn = sqlite3.connect("olist.db")

def run(sql):
    return pd.read_sql_query(sql, conn)

print("Running all queries...")

q1 = run("""
    SELECT category, COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(AVG(revenue), 2) AS avg_order_value,
        ROUND(SUM(revenue) * 100.0 / (SELECT SUM(revenue) FROM master), 1) AS revenue_pct
    FROM master WHERE category != 'Unknown'
    GROUP BY category ORDER BY total_revenue DESC LIMIT 15
""")

q2 = run("""
    WITH monthly AS (
        SELECT year, month, month_name, ROUND(SUM(revenue), 2) AS monthly_revenue
        FROM master GROUP BY year, month, month_name
    )
    SELECT year, month_name, monthly_revenue,
        ROUND((monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year, month))
        / LAG(monthly_revenue) OVER (ORDER BY year, month) * 100, 1) AS mom_growth_pct
    FROM monthly ORDER BY year, month
""")

q3 = run("""
    SELECT customer_state, COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(AVG(revenue), 2) AS avg_order_value,
        ROUND(AVG(delivery_days)) AS avg_delivery_days
    FROM master GROUP BY customer_state ORDER BY total_revenue DESC LIMIT 10
""")

q4 = run("""
    SELECT payment_type, COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(AVG(payment_installments), 1) AS avg_installments,
        ROUND(AVG(revenue), 2) AS avg_order_value
    FROM master WHERE payment_type != 'Not Defined'
    GROUP BY payment_type ORDER BY total_revenue DESC
""")

q5 = run("""
    WITH delivery_buckets AS (
        SELECT m.order_id, m.delivery_days, r.review_score,
            CASE
                WHEN m.delivery_days <= 7  THEN '1. Fast (7 days or less)'
                WHEN m.delivery_days <= 14 THEN '2. Normal (8 to 14 days)'
                WHEN m.delivery_days <= 21 THEN '3. Slow (15 to 21 days)'
                ELSE '4. Very slow (over 21 days)'
            END AS delivery_bucket
        FROM master m JOIN reviews r ON m.order_id = r.order_id
        WHERE m.delivery_days IS NOT NULL AND r.review_score IS NOT NULL
    )
    SELECT delivery_bucket, COUNT(*) AS total_orders,
        ROUND(AVG(review_score), 2) AS avg_review_score,
        ROUND(AVG(delivery_days), 1) AS avg_delivery_days
    FROM delivery_buckets GROUP BY delivery_bucket ORDER BY delivery_bucket
""")

q6 = run("""
    SELECT m.seller_id, s.seller_state,
        COUNT(DISTINCT m.order_id) AS total_orders,
        ROUND(SUM(m.revenue), 2) AS total_revenue,
        ROUND(AVG(m.revenue), 2) AS avg_order_value
    FROM master m JOIN sellers s ON m.seller_id = s.seller_id
    GROUP BY m.seller_id, s.seller_state
    ORDER BY total_revenue DESC LIMIT 10
""")

print("All queries done. Saving to Excel...")

with pd.ExcelWriter("olist_tableau_data.xlsx") as writer:
    q1.to_excel(writer, sheet_name="Revenue by Category",  index=False)
    q2.to_excel(writer, sheet_name="MoM Growth",           index=False)
    q3.to_excel(writer, sheet_name="Top States",           index=False)
    q4.to_excel(writer, sheet_name="Payment Types",        index=False)
    q5.to_excel(writer, sheet_name="Delivery vs Reviews",  index=False)
    q6.to_excel(writer, sheet_name="Top Sellers",          index=False)

print("Saved: olist_tableau_data.xlsx")
print("Ready for Tableau.")

conn.close()