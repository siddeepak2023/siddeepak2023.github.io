"""
01_data_pipeline.py
Pull 3 seasons of NBA game logs + today's games + betting odds.
Stores everything in SQLite at data/nba.db.

Run:  python3 01_data_pipeline.py
Set ODDS_API_KEY env var (or paste key below) to enable odds fetch.
"""

import os
import sys
import time
import sqlite3
import logging
import requests
import pandas as pd
from datetime import date, datetime
from nba_api.stats.endpoints import TeamGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams_static

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH       = "data/nba.db"
SEASONS       = ["2022-23", "2023-24", "2024-25"]
ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")   # export ODDS_API_KEY=xxx  or paste here
ODDS_SPORT    = "basketball_nba"
ODDS_REGION   = "us"
ODDS_MARKETS  = "h2h,spreads"
REQUEST_SLEEP = 1.2   # seconds between nba_api calls to avoid rate-limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Database schema ─────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS team_game_logs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            season        TEXT    NOT NULL,
            team_id       INTEGER NOT NULL,
            team_abbr     TEXT    NOT NULL,
            team_name     TEXT    NOT NULL,
            game_id       TEXT    NOT NULL,
            game_date     TEXT    NOT NULL,   -- 'YYYY-MM-DD'
            matchup       TEXT    NOT NULL,   -- 'LAL vs. BOS' or 'LAL @ BOS'
            is_home       INTEGER NOT NULL,   -- 1=home  0=away
            opponent_abbr TEXT    NOT NULL,
            wl            TEXT,               -- 'W' or 'L'
            win           INTEGER,            -- 1=win  0=loss
            season_w      INTEGER,            -- cumulative wins at this point
            season_l      INTEGER,            -- cumulative losses
            min           REAL,
            fgm REAL, fga REAL, fg_pct REAL,
            fg3m REAL, fg3a REAL, fg3_pct REAL,
            ftm  REAL, fta REAL, ft_pct REAL,
            oreb REAL, dreb REAL, reb REAL,
            ast  REAL, stl REAL, blk REAL,
            tov  REAL, pf  REAL, pts REAL,
            UNIQUE(game_id, team_id)
        );

        CREATE TABLE IF NOT EXISTS odds (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at    TEXT NOT NULL,
            game_id_api   TEXT,
            commence_time TEXT,
            home_team     TEXT NOT NULL,
            away_team     TEXT NOT NULL,
            home_ml       REAL,
            away_ml       REAL,
            home_spread   REAL,
            spread_price  REAL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id           TEXT NOT NULL,
            game_date         TEXT NOT NULL,
            home_team_abbr    TEXT NOT NULL,
            away_team_abbr    TEXT NOT NULL,
            pred_home_prob    REAL,
            implied_home_prob REAL,
            edge              REAL,
            recommendation    TEXT,
            model_version     TEXT,
            created_at        TEXT NOT NULL,
            UNIQUE(game_id, model_version)
        );

        CREATE INDEX IF NOT EXISTS idx_logs_game_id   ON team_game_logs(game_id);
        CREATE INDEX IF NOT EXISTS idx_logs_team_date ON team_game_logs(team_id, game_date);
        CREATE INDEX IF NOT EXISTS idx_logs_season    ON team_game_logs(season);
    """)
    conn.commit()
    log.info("Schema ready: %s", DB_PATH)


# ── nba_api helpers ─────────────────────────────────────────────────────────
def _safe_fetch(fn, retries=3, wait=6.0):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:
            log.warning("  Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(wait * attempt)
    return None


def _parse_is_home(matchup: str) -> int:
    """'LAL vs. BOS' → 1 (home)  |  'LAL @ BOS' → 0 (away)"""
    return 1 if "vs." in matchup else 0


def _parse_opponent(matchup: str, team_abbr: str) -> str:
    """Extract opponent abbreviation from matchup string."""
    clean = matchup.replace("vs.", "@")
    parts = [p.strip() for p in clean.split("@")]
    for p in parts:
        if p and p != team_abbr:
            return p
    return "UNK"


def _norm_date(raw: str) -> str:
    """Convert 'APR 01, 2024' or 'YYYY-MM-DD' to 'YYYY-MM-DD'."""
    raw = str(raw).strip()
    for fmt in ("%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def pull_team_season(conn, team_id: int, team_abbr: str,
                     team_name: str, season: str) -> int:
    """Pull one team's regular-season game log and upsert. Returns new rows inserted."""
    result = _safe_fetch(lambda: TeamGameLog(
        team_id=team_id,
        season=season,
        season_type_all_star="Regular Season",
        timeout=30,
    ))
    if result is None:
        return 0

    df = result.get_data_frames()[0]
    if df.empty:
        return 0

    # Normalise column names to upper-case
    df.columns = [c.upper() for c in df.columns]

    cur  = conn.cursor()
    inserted = 0

    for _, row in df.iterrows():
        is_home  = _parse_is_home(str(row.get("MATCHUP", "")))
        opp_abbr = _parse_opponent(str(row.get("MATCHUP", "")), team_abbr)
        wl       = str(row.get("WL", "")).strip().upper()
        win      = 1 if wl == "W" else (0 if wl == "L" else None)
        gdate    = _norm_date(row.get("GAME_DATE", ""))

        def _f(col, default=0.0):
            v = row.get(col, default)
            try:
                return float(v) if v is not None and str(v).strip() != "" else float(default)
            except (ValueError, TypeError):
                return float(default)

        try:
            cur.execute("""
                INSERT OR IGNORE INTO team_game_logs
                (season, team_id, team_abbr, team_name, game_id, game_date,
                 matchup, is_home, opponent_abbr, wl, win, season_w, season_l,
                 min, fgm, fga, fg_pct, fg3m, fg3a, fg3_pct,
                 ftm, fta, ft_pct, oreb, dreb, reb,
                 ast, stl, blk, tov, pf, pts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                season, int(team_id), team_abbr, team_name,
                str(row.get("GAME_ID", "")),
                gdate,
                str(row.get("MATCHUP", "")),
                is_home, opp_abbr, wl, win,
                int(_f("W")), int(_f("L")),
                _f("MIN"),
                _f("FGM"), _f("FGA"), _f("FG_PCT"),
                _f("FG3M"), _f("FG3A"), _f("FG3_PCT"),
                _f("FTM"),  _f("FTA"), _f("FT_PCT"),
                _f("OREB"), _f("DREB"), _f("REB"),
                _f("AST"),  _f("STL"), _f("BLK"),
                _f("TOV"),  _f("PF"),  _f("PTS"),
            ))
            inserted += cur.rowcount
        except sqlite3.Error as e:
            log.warning("Insert skip %s %s: %s", team_abbr, row.get("GAME_ID", ""), e)

    conn.commit()
    return inserted


# ── Today's games ────────────────────────────────────────────────────────────
def pull_todays_games() -> pd.DataFrame:
    today_str = date.today().strftime("%m/%d/%Y")
    result = _safe_fetch(lambda: ScoreboardV2(
        game_date=today_str,
        league_id="00",
        day_offset=0,
        timeout=30,
    ))
    if result is None:
        return pd.DataFrame()

    dfs = result.get_data_frames()
    header = dfs[0]  # GameHeader
    if header.empty:
        log.info("No games today (%s).", date.today())
        return pd.DataFrame()

    header.columns = [c.upper() for c in header.columns]
    log.info("Today's games (%d found):", len(header))
    for _, g in header.iterrows():
        log.info("  %s  vs  %s",
                 g.get("VISITOR_TEAM_CITY", "?"),
                 g.get("HOME_TEAM_CITY", "?"))
    return header


# ── Betting odds ─────────────────────────────────────────────────────────────
def pull_odds(conn) -> int:
    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY not set — skipping odds. "
                    "Set env var or paste key into script to enable.")
        return 0

    url = (
        f"https://api.the-odds-api.com/v4/sports/{ODDS_SPORT}/odds/"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions={ODDS_REGION}"
        f"&markets={ODDS_MARKETS}"
        f"&oddsFormat=american"
        f"&dateFormat=iso"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("Odds API error: %s", exc)
        return 0

    log.info("Odds API requests remaining: %s",
             resp.headers.get("x-requests-remaining", "?"))

    games   = resp.json()
    cur     = conn.cursor()
    inserted = 0
    now      = datetime.utcnow().isoformat()

    PREFERRED_BOOKS = {"draftkings", "fanduel", "betmgm", "bovada"}

    for game in games:
        home_team     = game.get("home_team", "")
        away_team     = game.get("away_team", "")
        commence_time = game.get("commence_time", "")
        game_id_api   = game.get("id", "")

        home_ml = away_ml = home_spread = spread_price = None

        bookmakers = game.get("bookmakers", [])
        # Prefer known liquid books
        ordered = sorted(bookmakers,
                         key=lambda b: 0 if b.get("key") in PREFERRED_BOOKS else 1)

        for bk in ordered:
            for market in bk.get("markets", []):
                if market["key"] == "h2h" and home_ml is None:
                    for o in market.get("outcomes", []):
                        if o["name"] == home_team:
                            home_ml = o["price"]
                        elif o["name"] == away_team:
                            away_ml = o["price"]
                elif market["key"] == "spreads" and home_spread is None:
                    for o in market.get("outcomes", []):
                        if o["name"] == home_team:
                            home_spread  = o.get("point")
                            spread_price = o.get("price")
            if home_ml is not None and home_spread is not None:
                break

        cur.execute("""
            INSERT INTO odds
            (fetched_at, game_id_api, commence_time, home_team, away_team,
             home_ml, away_ml, home_spread, spread_price)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (now, game_id_api, commence_time, home_team, away_team,
               home_ml, away_ml, home_spread, spread_price))
        inserted += 1
        log.info("  Odds — %s vs %s  |  ML: %s/%s  spread: %s",
                 away_team, home_team, away_ml, home_ml, home_spread)

    conn.commit()
    log.info("Inserted %d odds rows.", inserted)
    return inserted


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    all_teams = nba_teams_static.get_teams()
    log.info("NBA teams loaded: %d", len(all_teams))

    total_new = 0
    for season in SEASONS:
        log.info("═══ Season: %s ═══════════════════════════════", season)
        for i, t in enumerate(all_teams, 1):
            tid   = t["id"]
            abbr  = t["abbreviation"]
            name  = t["full_name"]

            # Skip if we already have a complete season (≥70 rows)
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM team_game_logs WHERE team_id=? AND season=?",
                (tid, season),
            )
            existing = cur.fetchone()[0]
            if existing >= 70:
                log.info("  [%2d/30] %-4s %s — cached (%d rows)", i, abbr, season, existing)
                continue

            log.info("  [%2d/30] %-4s %s — fetching ...", i, abbr, season)
            n = pull_team_season(conn, tid, abbr, name, season)
            total_new += n
            log.info("         → %d new rows inserted", n)
            time.sleep(REQUEST_SLEEP)

    log.info("Total new rows: %d", total_new)

    # Row counts by season
    cur = conn.cursor()
    cur.execute("SELECT season, COUNT(*) FROM team_game_logs GROUP BY season ORDER BY season")
    for row in cur.fetchall():
        log.info("  DB season %s: %d rows", row[0], row[1])

    # Today's games
    log.info("═══ Today's scoreboard ════════════════════════")
    today_df = pull_todays_games()
    if not today_df.empty:
        today_df.to_csv("data/todays_games.csv", index=False)
        log.info("Saved → data/todays_games.csv (%d games)", len(today_df))

    # Odds
    log.info("═══ Betting odds ══════════════════════════════")
    pull_odds(conn)

    conn.close()
    log.info("✓  Pipeline complete — %s", DB_PATH)


if __name__ == "__main__":
    main()
