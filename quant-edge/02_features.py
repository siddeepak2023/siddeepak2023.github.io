"""
02_features.py
Build cross-sectional feature set for every ticker × date.
Features: technical indicators, momentum, volatility, market context.
Target: 5-day forward return direction (1 = up, 0 = down/flat).

Run:  python3 02_features.py
"""

import sqlite3
import logging
import numpy as np
import pandas as pd

DB_PATH      = "data/quant.db"
FEATURES_CSV = "data/features.csv"
FWD_DAYS     = 5     # forward return horizon
MIN_ROWS     = 120   # minimum price history to include a ticker

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Technical indicator helpers ───────────────────────────────────────────────

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(series, period=20):
    sma   = series.rolling(period, min_periods=period).mean()
    std   = series.rolling(period, min_periods=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    width = (upper - lower) / sma.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pct_b, width


def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean() / close.replace(0, np.nan)


def realized_vol(returns, period=20):
    return returns.rolling(period, min_periods=period).std() * np.sqrt(252)


# ── Build features for one ticker ─────────────────────────────────────────────

def build_ticker_features(df):
    df = df.sort_values("date").copy()
    c  = df["adj_close"].replace(0, np.nan)
    h  = df["high"].replace(0, np.nan)
    l  = df["low"].replace(0, np.nan)
    v  = df["volume"].replace(0, np.nan)

    ret  = np.log(c / c.shift(1))

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["ret_1d"]  = ret
    df["ret_5d"]  = np.log(c / c.shift(5))
    df["ret_10d"] = np.log(c / c.shift(10))
    df["ret_20d"] = np.log(c / c.shift(20))
    df["ret_60d"] = np.log(c / c.shift(60))
    df["mom_accel"] = df["ret_5d"] - df["ret_20d"]   # short vs medium momentum
    df["price_52w_pct"] = c / c.rolling(252, min_periods=100).max()

    # ── Technical ─────────────────────────────────────────────────────────────
    df["rsi_14"]       = rsi(c, 14)
    df["rsi_7"]        = rsi(c,  7)
    ml, sl, hist       = macd(c)
    df["macd_hist"]    = hist / c.replace(0, np.nan)   # normalise by price
    df["macd_cross"]   = (ml > sl).astype(int)         # 1 = bullish cross
    df["bb_pct_b"],  df["bb_width"] = bollinger(c)
    df["atr_14"]       = atr(h, l, c, 14)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_avg = v.rolling(20, min_periods=10).mean()
    df["vol_ratio"]    = v / vol_avg.replace(0, np.nan)
    df["vol_trend"]    = v.rolling(5).mean() / vol_avg.replace(0, np.nan)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["vol_20d"]      = realized_vol(ret, 20)
    df["vol_60d"]      = realized_vol(ret, 60)
    df["vol_ratio_rv"] = df["vol_20d"] / df["vol_60d"].replace(0, np.nan)

    # ── Mean reversion ────────────────────────────────────────────────────────
    sma20  = c.rolling(20,  min_periods=10).mean()
    sma50  = c.rolling(50,  min_periods=25).mean()
    sma200 = c.rolling(200, min_periods=100).mean()
    df["pct_above_sma20"]  = (c - sma20)  / sma20.replace(0, np.nan)
    df["pct_above_sma50"]  = (c - sma50)  / sma50.replace(0, np.nan)
    df["pct_above_sma200"] = (c - sma200) / sma200.replace(0, np.nan)
    df["sma_20_50_cross"]  = (sma20 > sma50).astype(int)

    # ── Target: 5-day forward return direction ────────────────────────────────
    fwd_ret = np.log(c.shift(-FWD_DAYS) / c)
    df["fwd_ret_5d"] = fwd_ret
    df["target"]     = (fwd_ret > 0).astype(int)

    return df


# ── Attach market & sector context ────────────────────────────────────────────

def build_market_features(conn):
    """Return date-indexed df with SPY ret, VIX level, sector returns."""
    spy = pd.read_sql_query(
        "SELECT date, adj_close FROM prices WHERE ticker='SPY' ORDER BY date", conn)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date").sort_index()
    spy_ret = np.log(spy["adj_close"] / spy["adj_close"].shift(1))
    market = pd.DataFrame({
        "spy_ret_5d":  np.log(spy["adj_close"] / spy["adj_close"].shift(5)),
        "spy_ret_20d": np.log(spy["adj_close"] / spy["adj_close"].shift(20)),
    })

    # VIX
    vix = pd.read_sql_query(
        "SELECT date, adj_close FROM prices WHERE ticker='^VIX' ORDER BY date", conn)
    if not vix.empty:
        vix["date"] = pd.to_datetime(vix["date"])
        vix = vix.set_index("date")["adj_close"].rename("vix_close")
        vix_ma = vix.rolling(20, min_periods=10).mean()
        market["vix_level"]   = vix / vix_ma.replace(0, np.nan)   # normalised
        market["vix_ret_5d"]  = np.log(vix / vix.shift(5))

    # Sector ETF 5-day returns (attach later by ticker→sector)
    sector_etfs = pd.read_sql_query(
        "SELECT ticker, date, adj_close FROM prices WHERE is_etf=1 AND ticker NOT IN ('SPY','^VIX')"
        " ORDER BY date",
        pd.read_sql_query.__func__ if False else conn
    ) if False else pd.read_sql_query(
        "SELECT p.ticker, p.date, p.adj_close FROM prices p "
        "JOIN tickers t ON p.ticker=t.ticker "
        "WHERE t.is_etf=1 AND p.ticker NOT IN ('SPY','^VIX') ORDER BY p.date", conn)

    sector_map = {}
    for etf, grp in sector_etfs.groupby("ticker"):
        grp = grp.sort_values("date").set_index("date")
        grp.index = pd.to_datetime(grp.index)
        sector_map[etf] = np.log(grp["adj_close"] / grp["adj_close"].shift(5)).rename(f"sector_ret_5d_{etf}")

    sector_df = pd.concat(sector_map.values(), axis=1) if sector_map else pd.DataFrame()
    return market, sector_df


def get_sector_etf(sector, etf_map):
    mapping = {
        "Information Technology": "XLK",
        "Technology": "XLK",
        "Financials": "XLF",
        "Health Care": "XLV",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Materials": "XLB",
        "Real Estate": "XLRE",
        "Utilities": "XLU",
        "Communication Services": "XLC",
        "Communication": "XLC",
    }
    return mapping.get(sector, None)


# ── Cross-sectional z-score ───────────────────────────────────────────────────

NORM_COLS = [
    "ret_1d","ret_5d","ret_10d","ret_20d","ret_60d","mom_accel",
    "rsi_14","rsi_7","macd_hist","bb_pct_b","bb_width","atr_14",
    "vol_ratio","vol_trend","vol_20d","vol_60d","vol_ratio_rv",
    "pct_above_sma20","pct_above_sma50","pct_above_sma200",
    "vol_ratio","spy_ret_5d","spy_ret_20d","vix_level","vix_ret_5d",
    "sector_ret_5d",
]


def cross_section_zscore(df, cols):
    """Z-score each feature cross-sectionally (within each date)."""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        grp = out.groupby("date")[col]
        out[col] = grp.transform(lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else np.nan))
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(DB_PATH)

    log.info("Building market context features …")
    market_df, sector_df = build_market_features(conn)

    ticker_meta = pd.read_sql_query(
        "SELECT ticker, sector FROM tickers WHERE is_etf=0", conn)
    ticker_sector = dict(zip(ticker_meta.ticker, ticker_meta.sector))

    tickers = ticker_meta["ticker"].tolist()
    log.info("Processing %d tickers …", len(tickers))

    all_rows = []
    for i, ticker in enumerate(tickers):
        df = pd.read_sql_query(
            f"SELECT date, open, high, low, adj_close, volume FROM prices "
            f"WHERE ticker=? ORDER BY date", conn, params=(ticker,))
        if len(df) < MIN_ROWS:
            continue
        df["date"] = pd.to_datetime(df["date"])

        feats = build_ticker_features(df)
        feats["ticker"] = ticker
        feats["sector"] = ticker_sector.get(ticker, "Unknown")

        # Attach market context
        feats = feats.set_index("date")
        feats = feats.join(market_df, how="left")

        # Attach sector ETF return
        sector    = ticker_sector.get(ticker, "")
        etf_key   = get_sector_etf(sector, {})
        etf_col   = f"sector_ret_5d_{etf_key}" if etf_key else None
        if etf_col and etf_col in sector_df.columns:
            feats = feats.join(sector_df[[etf_col]].rename(
                columns={etf_col: "sector_ret_5d"}), how="left")
        else:
            feats["sector_ret_5d"] = np.nan

        feats = feats.reset_index()
        all_rows.append(feats)

        if (i + 1) % 50 == 0:
            log.info("  %d/%d done", i + 1, len(tickers))

    conn.close()

    log.info("Concatenating all ticker features …")
    combined = pd.concat(all_rows, ignore_index=True)
    combined["date"] = combined["date"].astype(str)

    # Drop rows missing target or key features
    key_cols = ["ret_5d","rsi_14","macd_hist","vol_20d","target"]
    before = len(combined)
    combined = combined.dropna(subset=key_cols)
    log.info("Dropped %d NaN rows → %d remaining", before - len(combined), len(combined))

    # Cross-sectional z-score
    log.info("Applying cross-sectional z-score …")
    combined = cross_section_zscore(combined, NORM_COLS)

    combined.to_csv(FEATURES_CSV, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", FEATURES_CSV, *combined.shape)

    # Stats
    log.info("Target distribution: %.1f%% up", combined["target"].mean() * 100)
    log.info("Date range: %s → %s", combined["date"].min(), combined["date"].max())
    log.info("Tickers: %d", combined["ticker"].nunique())
    log.info("✓  Feature engineering complete.")


if __name__ == "__main__":
    main()
