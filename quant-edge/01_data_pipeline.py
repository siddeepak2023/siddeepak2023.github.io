"""
01_data_pipeline.py
Pull S&P 500 price history + VIX + sector ETFs via yfinance.
Stores OHLCV in SQLite at data/quant.db.

Run:  python3 01_data_pipeline.py
"""

import sqlite3
import logging
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DB_PATH    = "data/quant.db"
YEARS_BACK = 6      # enough for walk-forward train/test splits
BATCH_SIZE = 50     # tickers per yfinance batch download

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker    TEXT NOT NULL,
            date      TEXT NOT NULL,
            open      REAL, high REAL, low REAL, close REAL,
            adj_close REAL, volume REAL,
            PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS tickers (
            ticker  TEXT PRIMARY KEY,
            name    TEXT,
            sector  TEXT,
            is_etf  INTEGER DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
        CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
    """)
    conn.commit()


def get_sp500_tickers():
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        import ssl, urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as resp:
            html = resp.read()
        tables = pd.read_html(html, header=0)
        df = tables[0]
        df.columns = [c.strip() for c in df.columns]
        symbol_col = next(c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower())
        sector_col = next(c for c in df.columns if "sector" in c.lower() or "gics" in c.lower())
        name_col   = next(c for c in df.columns if "security" in c.lower() or "name" in c.lower())
        df["ticker"] = df[symbol_col].str.replace(".", "-", regex=False).str.strip()
        df["sector"] = df[sector_col].str.strip()
        df["name"]   = df[name_col].str.strip()
        log.info("S&P 500 tickers loaded: %d", len(df))
        return df[["ticker", "name", "sector"]].drop_duplicates("ticker")
    except Exception as e:
        log.error("Failed to fetch S&P 500 list: %s", e)
        return pd.DataFrame(columns=["ticker", "name", "sector"])


def already_have(conn, ticker, min_rows=200):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prices WHERE ticker=?", (ticker,))
    return cur.fetchone()[0] >= min_rows


def flatten_df(df, ticker):
    """Flatten MultiIndex columns from yfinance and return simple OHLCV df."""
    if isinstance(df.columns, pd.MultiIndex):
        # New yfinance: columns are (Price, Ticker) — drop ticker level
        try:
            df = df.xs(ticker, axis=1, level="Ticker") if ticker in df.columns.get_level_values("Ticker") else df.droplevel(1, axis=1)
        except Exception:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def upsert_prices(conn, ticker, df):
    if df.empty:
        return 0
    df = flatten_df(df, ticker)
    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        return 0
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    col_map = {}
    for c in df.columns:
        if "open"   in c: col_map["open"]      = c
        if "high"   in c: col_map["high"]      = c
        if "low"    in c: col_map["low"]       = c
        if "close"  in c and "adj" not in c: col_map["close"] = c
        if "adj"    in c and "close" in c: col_map["adj_close"] = c
        if "volume" in c: col_map["volume"]    = c
    # yfinance auto_adjust=True: no adj_close column, use Close as adj_close
    if "adj_close" not in col_map and "close" in col_map:
        col_map["adj_close"] = col_map["close"]

    cur = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT OR REPLACE INTO prices
                (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                ticker,
                row[date_col],
                float(row.get(col_map.get("open", ""), 0) or 0),
                float(row.get(col_map.get("high", ""), 0) or 0),
                float(row.get(col_map.get("low",  ""), 0) or 0),
                float(row.get(col_map.get("close",""), 0) or 0),
                float(row.get(col_map.get("adj_close",""), 0) or 0),
                float(row.get(col_map.get("volume",""), 0) or 0),
            ))
            inserted += cur.rowcount
        except Exception:
            pass
    conn.commit()
    return inserted


def pull_batch(tickers, start, end):
    try:
        raw = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
            group_by="ticker"
        )
        return raw
    except Exception as e:
        log.warning("Batch download error: %s", e)
        return None


def main():
    import os
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * YEARS_BACK)).strftime("%Y-%m-%d")

    # ── S&P 500 constituents ─────────────────────────────────────────────────
    sp500 = get_sp500_tickers()

    # Upsert ticker metadata
    cur = conn.cursor()
    for _, row in sp500.iterrows():
        cur.execute("INSERT OR REPLACE INTO tickers (ticker, name, sector, is_etf) VALUES (?,?,?,0)",
                    (row.ticker, row.name, row.sector))
    conn.commit()

    # Batch download price history
    tickers = sp500["ticker"].tolist()
    total_new = 0

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = [t for t in tickers[i:i+BATCH_SIZE] if not already_have(conn, t)]
        if not batch:
            log.info("Batch %d–%d: all cached", i, i + BATCH_SIZE)
            continue

        log.info("Downloading batch %d–%d (%d tickers) …", i, i + BATCH_SIZE, len(batch))
        raw = pull_batch(batch, start, end)
        if raw is None or raw.empty:
            continue

        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                n = upsert_prices(conn, ticker, df)
                total_new += n
            except Exception as e:
                log.warning("  %s: %s", ticker, e)

        time.sleep(0.5)

    log.info("S&P 500 total new rows: %d", total_new)

    # ── VIX ──────────────────────────────────────────────────────────────────
    log.info("Downloading VIX …")
    cur.execute("INSERT OR REPLACE INTO tickers (ticker, name, sector, is_etf) VALUES (?,?,?,1)",
                ("^VIX", "CBOE VIX", "Market"))
    conn.commit()
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    n = upsert_prices(conn, "^VIX", vix)
    log.info("VIX: %d rows", n)

    # ── SPY (market benchmark) ───────────────────────────────────────────────
    log.info("Downloading SPY …")
    cur.execute("INSERT OR REPLACE INTO tickers (ticker, name, sector, is_etf) VALUES (?,?,?,1)",
                ("SPY", "SPDR S&P 500 ETF", "Market"))
    conn.commit()
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    n = upsert_prices(conn, "SPY", spy)
    log.info("SPY: %d rows", n)

    # ── Sector ETFs ───────────────────────────────────────────────────────────
    log.info("Downloading sector ETFs …")
    for etf, sector in SECTOR_ETFS.items():
        cur.execute("INSERT OR REPLACE INTO tickers (ticker, name, sector, is_etf) VALUES (?,?,?,1)",
                    (etf, f"{sector} Sector ETF", sector))
        conn.commit()
        df = yf.download(etf, start=start, end=end, auto_adjust=True, progress=False)
        n = upsert_prices(conn, etf, df)
        log.info("  %s: %d rows", etf, n)
        time.sleep(0.3)

    # Summary
    cur.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
    tcount = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM prices")
    rcount = cur.fetchone()[0]
    log.info("DB: %d tickers, %d price rows → %s", tcount, rcount, DB_PATH)
    conn.close()
    log.info("✓  Pipeline complete.")


if __name__ == "__main__":
    main()
