# Retail Customer Intelligence — RFM Segmentation, Churn Risk, Market Basket

Five-stage pandas/scikit-learn pipeline over the **UCI Online Retail II** dataset
(~805K transactions, Dec 2009 – Dec 2011, a UK gift-ware wholesaler). Produces customer
segments, a churn-risk model, association rules, and a single JSON payload that drives
[`retail_dashboard.html`](../retail_dashboard.html).

## Data — not committed

The raw dataset is a third-party download and is gitignored, along with every
intermediate it generates. To reproduce:

1. Download **Online Retail II** from the UCI Machine Learning Repository.
2. Place `online_retail_II.xlsx` in this directory. The scripts read the two sheets
   `Year 2009-2010` and `Year 2010-2011` by name (`01_cleaning_eda.py:12-13`).
3. `pip install -r requirements.txt`
4. Run the scripts in order. All paths are bare relative filenames, so run them from
   inside this directory.

```bash
python 01_cleaning_eda.py    # → online_retail_clean.csv, rfm_raw.csv, retail.db
python 02_segmentation.py    # → rfm_segmented.csv
python 03_churn_model.py     # → customer_features.csv, model_metrics.json
python 04_recommendations.py # → association_rules.json, segment_recommendations.json
python 05_export_dashboard.py# → dashboard_data.json
```

## Pipeline

| Script | What it does |
|---|---|
| `01_cleaning_eda.py` | Loads both sheets; drops cancellations, null CustomerIDs, non-positive quantity/price, and non-product stock codes; derives `Revenue = Quantity × Price`; builds an RFM table with `pd.qcut` 1–5 scores; writes a SQLite DB and runs one `LAG()` window query. |
| `02_segmentation.py` | `log1p` + `StandardScaler` on R/F/M, then K-Means. Adds 2-component PCA coordinates for the scatter plot. |
| `03_churn_model.py` | Labels churn temporally, builds 10 per-customer features from the pre-cutoff period only, fits a `RandomForestClassifier`, bins predicted probability into Low/Medium/High risk. |
| `04_recommendations.py` | Apriori market-basket mining on UK invoices; plus a per-segment revenue popularity ranking. |
| `05_export_dashboard.py` | Aggregation and serialisation only, no modelling. Compiles the 12-key `dashboard_data.json`. |

## Results

| Metric | Value |
|---|---|
| ROC-AUC (churn) | **0.824** |
| Accuracy | 75.4% |
| Base rate (majority class) | 55.3% — so accuracy alone overstates the lift; read the AUC |
| Customers modelled | 5,249 (4,199 train / 1,050 holdout) |
| Top feature | Recency, 27.4% importance |
| Total revenue in dataset | £17.74M |
| Revenue held by high-risk customers | £1.46M |

## Limitations — read these before quoting any number

These are stated because they materially change how the results should be read.

**£1.46M is exposure, not savings.** It is the sum of *already-earned, historical*
revenue belonging to the 2,290 customers whose predicted churn probability exceeds 0.65
(`05_export_dashboard.py:29`). It is not a forecast, not money recovered, not
incremental, and not a slice of the £17.74M total — the `Monetary` values behind it are
computed over Dec 2009 – Aug 2011 only, a different denominator.

**The evaluation holdout is random, not out-of-time.** The *labelling* is temporal —
features come strictly from before the 2011-09-01 cutoff and the label window is after
it, so there is no target leakage. But the train/test split is a random stratified 80/20
across customers (`03_churn_model.py:72`), not a second time-based cutoff. There is no
rolling-origin backtest. Metrics are therefore in-period.

**Risk tiers are partly in-sample.** `predict_proba` is applied to all 5,249 customers,
training rows included (`03_churn_model.py:91`), so roughly 80% of the risk scores — and
the £1.46M that derives from them — are fitted rather than held out.

**Recency partly restates the label.** Recency is the highest-importance feature and is
measured at the cutoff. A customer whose last purchase was long before the cutoff is
nearly by construction "churned," so the model is in part a re-expression of recency
rather than an independent signal.

**k=4 is a judgment call, not a fit statistic.** The script sweeps k=2..8 and prints
inertia and silhouette, but `K = 4` is assigned unconditionally
(`02_segmentation.py:41`). The comment states the criterion outright: business
interpretability. The four persona labels are hardcoded, so the pipeline always emits
exactly Champions / Loyal / At-Risk / Lost regardless of cluster shape.

**"Recommendations" are two different things.** The association rules are genuine Apriori
output, but scoped to UK invoices and the top 150 products only — a small slice of the
4,631-product catalogue. The per-segment lists are that segment's 8 highest-revenue
products: a popularity baseline, identical for every customer in the segment, not
personalisation.

**Recommendation quality is not evaluated.** No precision@k, no basket holdout, no
temporal split, no baseline comparison. Support, confidence and lift are descriptive
properties of the mined rules, not predictive accuracy.

**Thresholds are chosen, not tuned.** Risk cut points `[0, .35, .65, 1.0]`, `min_support
= 0.02`, `lift ≥ 1.5`, `confidence ≥ 0.30`, and the top-150 product cap are all
hardcoded with no sensitivity analysis.

**The dashboard is a frozen snapshot.** `dashboard_data.json` is inlined into
`retail_dashboard.html` as a `const DATA` literal — the page does not fetch. Re-running
script 05 updates the JSON but not the HTML, so the two can silently diverge.

All randomness is seeded (`random_state=42`), so runs are reproducible.

## What would make this stronger

1. A second time-based cutoff for evaluation, so the reported AUC is out-of-time.
2. Score only held-out customers before summing revenue-at-risk.
3. An ablation against a recency-only baseline, to show the other nine features earn
   their place.
4. Offline top-k evaluation of the association rules against a basket holdout.
5. Have the dashboard fetch the JSON instead of embedding it.
