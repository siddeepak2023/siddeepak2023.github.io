# Olist E-Commerce Analysis — SQL over 100K Brazilian Orders

Three-stage pandas + stdlib-SQLite pipeline over the **Olist Brazilian E-Commerce Public
Dataset** (Kaggle). Joins nine source CSVs into one master table, loads it into SQLite,
and runs six analytical queries — revenue by category, month-over-month growth with a
`LAG()` window function, state-level performance, payment mix, delivery-time vs review
score, and top sellers. Outputs an Excel workbook intended for Tableau, plus the static
[`olist_dashboard.html`](../olist_dashboard.html).

## Data — not committed

The nine input CSVs are a third-party Kaggle download and are gitignored, along with
every artifact they generate (`olist_master_clean.csv` alone is 36 MB). To reproduce:

1. Download the **Olist Brazilian E-Commerce Public Dataset** from Kaggle.
2. Put these nine files in this directory, unrenamed:
   `olist_orders_dataset.csv`, `olist_order_items_dataset.csv`,
   `olist_order_payments_dataset.csv`, `olist_order_reviews_dataset.csv`,
   `olist_products_dataset.csv`, `olist_customers_dataset.csv`,
   `olist_sellers_dataset.csv`, `olist_geolocation_dataset.csv`,
   `product_category_name_translation.csv`
3. `pip install -r requirements.txt`
4. Run in order, from inside this directory (all paths are bare relative filenames):

```bash
python 01_cleaning.py        # → olist_master_clean.csv, olist.db (7 tables)
python 02_sql_fix.py         # → adds the reviews table to olist.db, writes q6/q7 xlsx
python 03_export_tableau.py  # → olist_tableau_data.xlsx (6 sheets)
```

Run `02_sql_fix.py` before `03_export_tableau.py` — script 01 does not load the reviews
table, and the delivery-vs-review query needs it.

## Scope

Filtered to `order_status == "delivered"` only (`01_cleaning.py:52`): **96,478 orders,
R$15.4M revenue, Sep 2016 – Aug 2018, 27 states, 73 product categories.** Revenue is
defined as `price + freight_value` per order item (`01_cleaning.py:74`).

## Known defects — unfixed, and they affect published numbers

Stated plainly because the numbers in the Excel export and `top_states.csv` are wrong in
a specific, describable way.

**The master table is at order-item grain, not order grain.** `01_cleaning.py:117-119`
left-joins orders to order_items, and one order can have many items, so
`olist_master_clean.csv` holds 110,197 rows for 96,478 distinct orders — about 14% row
inflation. Script 01 prints that row count as "Master table: … rows" without noting the
grain change.

Consequences, all still present in the current code:

- **`avg_order_value` is average revenue per *item row*, not per order**
  (`03_export_tableau.py:14`, `:35`, `:70`, `02_sql_fix.py:54`). Every `avg_order_value`
  figure in `olist_tableau_data.xlsx` and `top_states.csv` is understated by roughly the
  mean items-per-order factor. Example: the export reports São Paulo at R$124.22; the
  true order-grain AOV is R$142.46.
- **The delivery-bucket counts are inflated and mislabelled.**
  `02_sql_fix.py:32-33` joins master to reviews, so the result is one row per
  (order-item × review) — two compounding fan-outs, since some orders carry multiple
  reviews. `COUNT(*) AS total_orders` then sums to 110,005 against only 96,478 delivered
  orders.
- **`avg_review_score` is item-weighted.** A 12-item order counts twelve times toward the
  mean. Same for `AVG(delivery_days)` per state (`:35`) and
  `AVG(payment_installments)` (`:41`).
- `COUNT(DISTINCT order_id)` **is** correct — order counts in the five queries that use
  it are trustworthy.

**Correct fix:** deduplicate to order grain before aggregating (or aggregate order_items
to one row per order first), then recompute. Not yet done.

**There is no statistical model in this project.** No regression, no correlation
coefficient, no significance test — the entire analysis is SQL `SUM`/`AVG`/`COUNT`/`LAG`
and pandas `groupby`. Any correlation or causal claim about delivery speed and reviews
would be unsupported by this code; the pipeline produces only a bucketed cross-tab. The
dashboard has been corrected to describe it as an observed pattern, not a driver.

**`02_sql_fix.py` fixed a missing table, not a bug in the analysis.** Script 01 writes
seven tables and omits `reviews`, so review queries failed; script 02 loads it and
re-runs those two queries. It changes no join logic and no grain.

**Loose ends in the code:** `olist_geolocation_dataset.csv` is read
(`01_cleaning.py:16`) and never used, so "9 tables joined" is really 9 read and 8 in the
database. `01_cleaning.py:157` tells you to run `02_sql_analysis.py`, which does not
exist in this repo — the file is `02_sql_fix.py`, and the original 7-query script is
missing. `02_sql_fix.py:69` prints "All 7 queries complete" while running two.
`top_states.csv` is committed but no script writes it; it appears to be a manual export.
`numpy` and `os` are imported in script 01 and never used.

## What would make this stronger

1. Aggregate to order grain before any `AVG`/`COUNT(*)`, then regenerate every output.
2. Recover or rewrite the missing `02_sql_analysis.py` so the pipeline is complete.
3. If the delivery/review relationship is the headline, model it properly — order-grain,
   with state and category controls, and report an effect size with its uncertainty.
4. Use `customer_unique_id` rather than `customer_id` if repeat-purchase behaviour is
   ever analysed; `customer_id` is per-order in this dataset and cannot express repeats.
