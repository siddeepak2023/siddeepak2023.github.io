# Quant Edge — Cross-Sectional Equity Ranking (S&P 500, 5-Day Horizon)

Walk-forward machine-learning pipeline that ranks S&P 500 constituents on 5-day forward
direction. Real market data from `yfinance`, 28 engineered features standardised
cross-sectionally per date, gradient-boosted classifier with isotonic calibration,
evaluated on a 10-day-embargoed walk-forward plus a held-out final six months.

**Headline result: an out-of-time holdout AUC of 0.5047 — this model has no predictive
edge.** That is the honest finding, and it is what the project should be read for. The
engineering below is sound; the signal is not there.

## Pipeline

```bash
pip install -r requirements.txt
python 01_data_pipeline.py   # yfinance → SQLite. 503 tickers + SPY + ^VIX + 11 sector ETFs, 6y
python 02_features.py        # 28 features, per-date cross-sectional z-scores, 5d fwd label
python 03_model.py           # Optuna search → walk-forward backtest → final holdout eval
python 04_screener.py        # today's ranked signals
python 05_export_dashboard.py# → data/*.json for quant_edge_dashboard.html
```

## What is defensible

| Result | Value | Why it holds |
|---|---|---|
| **Holdout AUC** | **0.5047** | Final 6 months, split at `03_model.py:233-236`, never seen by the tuning search |
| Holdout accuracy | 49.66% | Same clean holdout |
| Brier score | 0.2534 | Same |
| Walk-forward fold AUCs | 0.4682–0.5511, mean 0.5128 | 12 folds, 3-year rolling train, 3-month test, **10-day embargo** (`03_model.py:35`) that correctly exceeds the 5-day label horizon |
| Universe / span / features | 503 names, 5.92 years, 28 features | Factual |

The architecture is the part worth defending: time-aware splits with an embargo longer
than the label window, per-date cross-sectional normalisation
(`02_features.py:199-207` — grouped by date, so no cross-date leakage), isotonic
calibration, and a label that is strictly forward-looking (`02_features.py:117-119`).

## What is NOT defensible — and is withheld from the dashboard

The Sharpe, total-return and max-drawdown figures are no longer displayed. Four
independent defects, any one of which would disqualify them:

**1. The equity curve counts every return five times.** `fwd_ret_5d` is a 5-day forward
return, but `03_model.py:167,181` groups by calendar date and calls `.cumprod()`, so ~756
overlapping 5-day returns compound as if they were 756 independent holding periods.

**2. `total_return` was never actually computed.** The statistic at `03_model.py:211`
evaluated to `NaN` — it is literally `NaN` in `data/model_metrics.json`. A fallback at
`05_export_dashboard.py:206-213` then back-filled the value from the last point of the
plotted curve, which is how `663.90%` reached the page. The failure was never surfaced.

**3. Zero trading frictions.** No commission, spread, slippage, market impact or borrow
cost anywhere in the repository, against roughly 200 entries per day across 151,432
trades. Entry is assumed at the same closing price from which every feature was computed
— no next-open, no delay. Position sizing is equal-weight with no cap, and because a
fresh full-weight book opens daily while positions are held five days, the
implementation implies roughly 5× gross exposure with no financing cost charged.

**4. Sharpe uses an inconsistent period convention.** `03_model.py:171-173` annualises at
`252/5` periods on a series with ~252 observations per year, and the naive standard
deviation ignores the serial correlation that overlapping windows guarantee. A Sharpe of
0.977 alongside a −47.7% drawdown is arithmetically incompatible with the return figure
it sat next to.

Corrected for the overlap alone, the same data returns roughly **+50%** over the 36-month
window against SPY's **+76%** from the pipeline's own benchmark series. The strategy loses
to buy-and-hold before a single basis point of cost.

## Two further biases that affect everything above

**Survivorship.** `01_data_pipeline.py:61-85` reads *today's* S&P 500 membership from
Wikipedia and downloads six years of history for exactly those 503 current members. There
is no point-in-time constituent handling, so every deletion — bankruptcy, acquisition,
market-cap collapse — is invisible to the 2020–2026 backtest.

**Tuning contamination.** Optuna selects hyperparameters by maximising AUC over the whole
`train_pool` (`03_model.py:238-263`), and `walk_forward_backtest` is then run on the full
dataset with those parameters. Eleven of twelve folds test on dates inside the
optimisation pool. No P&L statistic in this project has ever been measured on data that
did not inform model selection. The holdout is used only for AUC, accuracy, Brier and
permutation importance — never for a return figure.

Related: the reported CV AUC of 0.518 is the best of 50 Optuna trials with no correction
for selection, `TimeSeriesSplit` at `03_model.py:243` has no embargo (so training and
test rows share four of five label days), and `optuna.create_study()` is called without a
sampler seed, so `best_params` is not reproducible across runs.

## Known smaller defects

- `04_screener.py:171` writes a field named `rsi_14` whose values have already been
  cross-sectionally z-scored at `:96` — `data/screener.json` therefore contains z-scores
  labelled as RSI. Not rendered, but wrong in a published artifact.
- Strategy A cannot short. The frame is filtered to `prob >= 0.60` and `pred` is
  therefore always 1, making the `else -fwd_ret_5d` branch dead code. The dashboard's
  "Strong Sell (≤35%)" tier is displayed but never traded.
- `fwd_ret_5d` is a log return but is compounded as a simple return.
- "21-Day Sector Performance" uses `Timedelta(days=21)` — 21 calendar days, ~14 trading
  days — and scales arithmetically rather than geometrically.
- `05_export_dashboard.py:123` carries a stale comment reading "Synthetic decile table";
  the function body returns real per-fold AUCs. Nothing synthetic is generated anywhere.
- "Signals Today" is dated from the last pipeline run, not the current date.

## To make the backtest quotable

1. De-overlap: hold to the label horizon, or sample every 5th day, before compounding.
2. Charge commission and spread, and execute at the next open rather than the decision
   close.
3. Rebuild the universe point-in-time from historical index membership.
4. Confirm every P&L statistic on the untouched holdout, not on the tuning span.
5. Purge and embargo the Optuna CV as well, and seed the sampler.
6. State the Sharpe convention explicitly and adjust for serial correlation.

Until those are done, the AUC is the result and the return figures are not.
