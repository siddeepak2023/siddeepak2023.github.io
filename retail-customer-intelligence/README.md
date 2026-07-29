# Retail Customer Intelligence — RFM Segmentation, Churn Risk, Market Basket

Five-stage pandas/scikit-learn pipeline over the **UCI Online Retail II** dataset
(802,712 transactions after cleaning, Dec 2009 – Dec 2011, a UK gift-ware wholesaler). Produces customer
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

Regenerated against the canonical UCI Online Retail II file. An earlier version of this
README published 0.824 / £17.74M / £1.46M; those figures were not reproducible from the
public dataset and are superseded — see *Limitations*.

| Metric | Value |
|---|---|
| ROC-AUC (churn), held-out | **0.796** |
| Accuracy, held-out | 73.3% |
| Base rate (majority class) | 55.3% — accuracy alone overstates the lift; read the AUC |
| Customers modelled | 5,233 (4,186 train / 1,047 holdout) |
| Total revenue in dataset | £17.45M |
| Transactions after cleaning | 802,712 |
| Countries | 41 |
| precision@k, top 100 contacts | 90.0% |

## Limitations — read these before quoting any number

These are stated because they materially change how the results should be read.

**£1.46M is retired, not restated.** The old figure summed *already-earned, historical*
revenue for customers scored above 0.65 — and scored them with a model that had trained on
80% of them. It was an in-sample total of money already booked, described as money at risk.
`06_retention_economics.py` replaces it: held-out customers only, model refit on the training
split, value defined as forward revenue observed *after* the cutoff. The artifact records the
supersession (`retention_economics.json → supersedes.old_figure_gbp`). Sizing now comes from a
precision@k curve — 90.0% of the top 100 are real churners, 78.0% of the top 500 — and the
useful result is that median forward value is ~176× the £3 contact cost, so **capacity, not
cost, is the binding constraint.** The save rate is assumed, not measured.

**The evaluation holdout is random, not out-of-time.** The *labelling* is temporal —
features come strictly from before the 2011-09-01 cutoff and the label window is after
it, so there is no target leakage. But the train/test split is a random stratified 80/20
across customers (`03_churn_model.py:72`), not a second time-based cutoff. There is no
rolling-origin backtest. Metrics are therefore in-period.

**Risk tiers in the legacy dashboard payload are partly in-sample.** `03_churn_model.py:91`
applies `predict_proba` to all customers, training rows included. That is why the retention
economics are computed separately in `06_retention_economics.py`, which scores the holdout only.

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
