"""
04_screener.py
Generate today's model predictions for every S&P 500 ticker.
Outputs data/screener.json with ranked picks, sector breakdown, and signals.

Run:  python3 04_screener.py
"""

import json
import logging
import pickle
import sqlite3
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

DB_PATH      = "data/quant.db"
MODEL_PKL    = "data/model.pkl"
SCREENER_JSON = "data/screener.json"
CONF_THRESHOLD = 0.60

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_model():
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]


def _load_features_module():
    """Load 02_features.py via importlib (filename starts with digit)."""
    import importlib.util, os
    path = os.path.join(os.path.dirname(__file__), "02_features.py")
    spec = importlib.util.spec_from_file_location("features_mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_latest_features(conn, feat_cols):
    """Build features for the most recent available date for each ticker."""
    fm = _load_features_module()
    build_ticker_features = fm.build_ticker_features
    build_market_features  = fm.build_market_features
    get_sector_etf         = fm.get_sector_etf
    NORM_COLS              = fm.NORM_COLS

    market_df, sector_df = build_market_features(conn)

    ticker_meta = pd.read_sql_query(
        "SELECT ticker, sector FROM tickers WHERE is_etf=0", conn)
    ticker_sector = dict(zip(ticker_meta.ticker, ticker_meta.sector))

    rows = []
    for ticker in ticker_meta["ticker"].tolist():
        df = pd.read_sql_query(
            "SELECT date, open, high, low, adj_close, volume FROM prices "
            "WHERE ticker=? ORDER BY date", conn, params=(ticker,))
        if len(df) < 80:
            continue
        df["date"] = pd.to_datetime(df["date"])
        feats = build_ticker_features(df)
        feats["ticker"] = ticker
        feats["sector"] = ticker_sector.get(ticker, "Unknown")
        feats = feats.set_index("date")
        feats = feats.join(market_df, how="left")

        sector    = ticker_sector.get(ticker, "")
        etf_key   = get_sector_etf(sector, {})
        etf_col   = f"sector_ret_5d_{etf_key}" if etf_key else None
        if etf_col and etf_col in sector_df.columns:
            feats = feats.join(sector_df[[etf_col]].rename(
                columns={etf_col: "sector_ret_5d"}), how="left")
        else:
            feats["sector_ret_5d"] = np.nan

        # Take latest complete row
        latest = feats.dropna(subset=["ret_5d","rsi_14","vol_20d"]).tail(1)
        if latest.empty:
            continue
        latest = latest.reset_index().rename(columns={"index": "date"})
        rows.append(latest)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)

    # Cross-sectional z-score on latest date
    cross_section_zscore = fm.cross_section_zscore
    combined = cross_section_zscore(combined, NORM_COLS)
    return combined


def get_price_info(conn, ticker):
    df = pd.read_sql_query(
        "SELECT date, adj_close FROM prices WHERE ticker=? ORDER BY date DESC LIMIT 30",
        conn, params=(ticker,))
    if df.empty:
        return {}
    df = df.sort_values("date")
    latest = df.iloc[-1]
    prev5  = df.iloc[-6]["adj_close"] if len(df) >= 6 else df.iloc[0]["adj_close"]
    return {
        "price":   round(float(latest["adj_close"]), 2),
        "ret_5d":  round(float((latest["adj_close"] / prev5 - 1) * 100), 2),
        "date":    str(latest["date"]),
        "sparkline": df["adj_close"].round(2).tolist()[-20:],
    }


def main():
    log.info("Loading model …")
    model, feat_cols = load_model()

    conn = sqlite3.connect(DB_PATH)

    log.info("Building latest features for all tickers …")
    try:
        latest_df = get_latest_features(conn, feat_cols)
    except Exception as e:
        log.error("Feature build failed: %s — falling back to CSV", e)
        # Fallback: use last row per ticker from features.csv
        df_all = pd.read_csv("data/features.csv")
        df_all["date"] = pd.to_datetime(df_all["date"])
        latest_df = df_all.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)

    if latest_df.empty:
        log.error("No feature data available.")
        conn.close()
        return

    available = [c for c in feat_cols if c in latest_df.columns]
    X = latest_df[available].values
    probs = model.predict_proba(X)[:, 1]
    latest_df["prob_up"] = probs
    latest_df["signal"]  = (probs >= CONF_THRESHOLD).astype(int)

    # Attach price info
    price_info = {}
    tickers = latest_df["ticker"].tolist()
    for t in tickers:
        price_info[t] = get_price_info(conn, t)
    conn.close()

    latest_df["price"]    = latest_df["ticker"].map(lambda t: price_info.get(t, {}).get("price", None))
    latest_df["ret_5d_pct"] = latest_df["ticker"].map(lambda t: price_info.get(t, {}).get("ret_5d", None))
    latest_df["sparkline"]  = latest_df["ticker"].map(lambda t: price_info.get(t, {}).get("sparkline", []))

    # Sort by probability
    latest_df = latest_df.sort_values("prob_up", ascending=False).reset_index(drop=True)
    latest_df["rank"] = latest_df.index + 1

    # Build output
    picks = []
    for _, row in latest_df.iterrows():
        picks.append({
            "rank":       int(row["rank"]),
            "ticker":     row["ticker"],
            "sector":     row.get("sector", "Unknown"),
            "prob_up":    round(float(row["prob_up"]), 4),
            "signal":     int(row["signal"]),
            "price":      row.get("price"),
            "ret_5d_pct": row.get("ret_5d_pct"),
            "sparkline":  row.get("sparkline", []),
            "rsi_14":     round(float(row.get("rsi_14_raw", row.get("rsi_14", 50))), 1) if "rsi_14" in row else None,
            "vol_20d":    round(float(row.get("vol_20d", 0)), 4) if "vol_20d" in row else None,
        })

    # Sector breakdown of top signals
    signals_df = latest_df[latest_df["signal"] == 1]
    sector_counts = signals_df["sector"].value_counts().to_dict() if not signals_df.empty else {}

    # Conviction tiers
    strong_buys  = latest_df[latest_df["prob_up"] >= 0.65]
    buys         = latest_df[(latest_df["prob_up"] >= 0.60) & (latest_df["prob_up"] < 0.65)]
    strong_sells = latest_df[latest_df["prob_up"] <= 0.35]

    out = {
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "date":            date.today().isoformat(),
        "n_tickers":       len(picks),
        "n_signals":       int(latest_df["signal"].sum()),
        "conf_threshold":  CONF_THRESHOLD,
        "avg_prob":        round(float(latest_df["prob_up"].mean()), 4),
        "picks":           picks,
        "sector_signal_counts": sector_counts,
        "tiers": {
            "strong_buy":  strong_buys["ticker"].tolist(),
            "buy":         buys["ticker"].tolist(),
            "strong_sell": strong_sells["ticker"].tolist(),
        },
    }

    with open(SCREENER_JSON, "w") as f:
        json.dump(out, f, indent=2)

    log.info("Screener → %s", SCREENER_JSON)
    log.info("  %d tickers scored, %d signals (≥%.0f%% prob)",
             len(picks), out["n_signals"], CONF_THRESHOLD * 100)
    log.info("  Strong buys: %d  |  Strong sells: %d",
             len(strong_buys), len(strong_sells))
    log.info("✓  Screener complete.")


if __name__ == "__main__":
    main()
