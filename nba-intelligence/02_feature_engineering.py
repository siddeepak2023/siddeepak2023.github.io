"""
02_feature_engineering.py
Build multi-window rolling features + Elo ratings + defensive rating for every game.
Outputs data/features.csv with one row per (game_id, home_team, away_team).

Improvements over v1:
  - 9 seasons (2016-17 → 2024-25) instead of 3
  - Elo ratings (pre-game, updated after every result)
  - Defensive rating (opponent pts per 100 possessions)
  - Net rating = off_rtg - def_rtg
  - Multiple rolling windows: 5, 10, 20 games
  - Exponentially-weighted moving average (span=10)
  - Back-to-back flag (rest_days <= 1)
  - Bubble flag for 2019-20 restart games (no home advantage)

Run:  python3 02_feature_engineering.py
"""

import sqlite3
import logging
import numpy as np
import pandas as pd

DB_PATH      = "data/nba.db"
FEATURES_CSV = "data/features.csv"
ELO_K        = 20
ELO_BASE     = 1500
BUBBLE_START = "2020-07-30"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Load raw logs ─────────────────────────────────────────────────────────────
def load_logs(conn):
    df = pd.read_sql_query("""
        SELECT a.season, a.team_id, a.team_abbr, a.game_id, a.game_date,
               a.is_home, a.opponent_abbr, a.win, a.wl,
               a.pts, a.fga, a.fgm, a.fg3m, a.fg3a, a.ftm, a.fta,
               a.oreb, a.dreb, a.reb, a.ast, a.stl, a.blk, a.tov, a.pf, a.min,
               b.pts AS opp_pts
        FROM team_game_logs a
        LEFT JOIN team_game_logs b
          ON a.game_id = b.game_id AND a.team_id != b.team_id
        ORDER BY a.team_id, a.game_date
    """, conn)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ── Elo ratings ───────────────────────────────────────────────────────────────
def compute_elo(df):
    """Return dict of game_id → {home_pre_elo, away_pre_elo, elo_diff}."""
    elo = {}
    records = {}

    home_games = (df[df["is_home"] == 1]
                  .dropna(subset=["win"])
                  [["game_id", "game_date", "team_abbr", "opponent_abbr", "win"]]
                  .sort_values("game_date")
                  .copy())

    for _, row in home_games.iterrows():
        ht, at = row["team_abbr"], row["opponent_abbr"]
        h_elo = elo.get(ht, ELO_BASE)
        a_elo = elo.get(at, ELO_BASE)

        records[row["game_id"]] = {
            "home_pre_elo": round(h_elo, 1),
            "away_pre_elo": round(a_elo, 1),
            "elo_diff":     round(h_elo - a_elo, 1),
        }

        exp_home = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
        actual   = float(row["win"])
        elo[ht]  = h_elo + ELO_K * (actual - exp_home)
        elo[at]  = a_elo + ELO_K * ((1 - actual) - (1 - exp_home))

    return records


# ── Per-team rolling features ─────────────────────────────────────────────────
def _possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + 0.4 * fta

def _ts(pts, fga, fta):
    d = 2 * (fga + 0.44 * fta)
    return pts / d if d > 0 else 0.0


def build_team_features(df):
    records = []
    for team_id, grp in df.groupby("team_id"):
        grp = grp.sort_values("game_date").reset_index(drop=True)

        grp["poss"]    = grp.apply(lambda r: _possessions(r.fga, r.oreb, r.tov, r.fta), axis=1)
        grp["off_rtg"] = (grp["pts"] / grp["poss"].replace(0, np.nan)) * 100
        grp["def_rtg"] = (grp["opp_pts"].fillna(0) / grp["poss"].replace(0, np.nan)) * 100
        grp["net_rtg"] = grp["off_rtg"] - grp["def_rtg"]
        grp["ts"]      = grp.apply(lambda r: _ts(r.pts, r.fga, r.fta), axis=1)
        grp["pace"]    = grp["poss"]
        grp["rest_days"] = grp["game_date"].diff().dt.days.fillna(3).clip(upper=10)
        grp["is_b2b"]  = (grp["rest_days"] <= 1).astype(int)
        grp["is_bubble"] = (
            (grp["season"] == "2019-20") &
            (grp["game_date"] >= pd.Timestamp(BUBBLE_START))
        ).astype(int)

        stat_cols = ["win", "pts", "off_rtg", "def_rtg", "net_rtg", "ts", "pace",
                     "tov", "oreb", "dreb", "reb", "ast", "fga", "fgm",
                     "fg3m", "fg3a", "stl", "blk"]
        shifted = grp[stat_cols].shift(1)

        # Three rolling windows
        roll5  = shifted.rolling(5,  min_periods=3).mean()
        roll10 = shifted.rolling(10, min_periods=5).mean()
        roll20 = shifted.rolling(20, min_periods=8).mean()
        # Exponentially-weighted (more weight on recent games)
        ewm10  = shifted.ewm(span=10, min_periods=5).mean()

        for i, row in grp.iterrows():
            if i < 5:
                continue
            r5, r10, r20, ew = roll5.loc[i], roll10.loc[i], roll20.loc[i], ewm10.loc[i]
            fga5 = r5.get("fga", 0) or 0
            fga10 = r10.get("fga", 0) or 0
            fga20 = r20.get("fga", 0) or 0

            rec = {
                "game_id":    row.game_id,
                "game_date":  row.game_date,
                "season":     row.season,
                "team_id":    team_id,
                "team_abbr":  row.team_abbr,
                "is_home":    int(row.is_home),
                "opponent_abbr": row.opponent_abbr,
                "win":        row.win,
                # ── 10-game rolling (primary) ──
                "roll_win_pct":  r10["win"],
                "roll_pts":      r10["pts"],
                "roll_off_rtg":  r10["off_rtg"],
                "roll_def_rtg":  r10["def_rtg"],
                "roll_net_rtg":  r10["net_rtg"],
                "roll_ts":       r10["ts"],
                "roll_pace":     r10["pace"],
                "roll_tov":      r10["tov"],
                "roll_oreb":     r10["oreb"],
                "roll_reb":      r10["reb"],
                "roll_ast":      r10["ast"],
                "roll_stl":      r10["stl"],
                "roll_blk":      r10["blk"],
                "roll_fg3_rate": (r10["fg3a"] / fga10) if fga10 > 0 else np.nan,
                # ── 5-game (hot streak) ──
                "roll5_win_pct": r5["win"],
                "roll5_net_rtg": r5["net_rtg"],
                "roll5_pts":     r5["pts"],
                # ── 20-game (season trend) ──
                "roll20_win_pct": r20["win"],
                "roll20_net_rtg": r20["net_rtg"],
                "roll20_fg3_rate": (r20["fg3a"] / fga20) if fga20 > 0 else np.nan,
                # ── EWMA ──
                "ewm_win_pct":  ew["win"],
                "ewm_net_rtg":  ew["net_rtg"],
                "ewm_pts":      ew["pts"],
                # ── Context ──
                "rest_days":     row.rest_days,
                "is_b2b":        row.is_b2b,
                "is_bubble":     row.is_bubble,
                "season_win_pct": (grp.loc[:i-1, "win"].sum() / max(i, 1)),
            }
            records.append(rec)

    return pd.DataFrame(records)


# ── Pair home/away into one game row ─────────────────────────────────────────
def build_game_features(team_feats, elo_map):
    home = team_feats[team_feats["is_home"] == 1].copy()
    away = team_feats[team_feats["is_home"] == 0].copy()

    skip = {"game_id", "game_date", "season"}
    home = home.rename(columns={c: f"home_{c}" for c in team_feats.columns if c not in skip})
    away = away.rename(columns={c: f"away_{c}" for c in team_feats.columns if c not in skip})

    merged = home.merge(away, on=["game_id", "game_date", "season"], how="inner")

    # Elo columns
    merged["home_pre_elo"] = merged["game_id"].map(lambda g: elo_map.get(g, {}).get("home_pre_elo", ELO_BASE))
    merged["away_pre_elo"] = merged["game_id"].map(lambda g: elo_map.get(g, {}).get("away_pre_elo", ELO_BASE))
    merged["elo_diff"]     = merged["game_id"].map(lambda g: elo_map.get(g, {}).get("elo_diff", 0))

    # Differential features
    diff_cols = [
        "roll_win_pct", "roll_pts", "roll_off_rtg", "roll_def_rtg", "roll_net_rtg",
        "roll_ts", "roll_pace", "roll_tov", "roll_oreb", "roll_reb", "roll_ast",
        "roll_fg3_rate", "roll5_win_pct", "roll5_net_rtg", "roll20_win_pct",
        "roll20_net_rtg", "ewm_win_pct", "ewm_net_rtg", "ewm_pts",
        "rest_days", "season_win_pct",
    ]
    for c in diff_cols:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in merged.columns and ac in merged.columns:
            merged[f"diff_{c}"] = merged[hc] - merged[ac]

    merged["diff_is_b2b"]   = merged["home_is_b2b"]   - merged["away_is_b2b"]
    merged["diff_elo"]      = merged["elo_diff"]
    merged["home_win"]      = merged["home_win"].astype(int)

    return merged


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    conn = sqlite3.connect(DB_PATH)
    log.info("Loading game logs (with opponent pts via self-join) …")
    raw = load_logs(conn)
    conn.close()
    log.info("Loaded %d team-game rows across %d teams over %d seasons",
             len(raw), raw["team_id"].nunique(), raw["season"].nunique())

    log.info("Computing Elo ratings …")
    elo_map = compute_elo(raw)
    log.info("Elo map: %d games", len(elo_map))

    log.info("Computing multi-window rolling features …")
    team_feats = build_team_features(raw)
    log.info("Team feature rows: %d", len(team_feats))

    log.info("Pairing home/away into game rows …")
    game_feats = build_game_features(team_feats, elo_map)
    log.info("Game feature rows: %d", len(game_feats))

    feature_cols = [c for c in game_feats.columns
                    if c.startswith(("home_roll", "away_roll", "diff_",
                                     "home_ewm", "away_ewm",
                                     "home_pre_elo", "away_pre_elo", "elo_diff",
                                     "home_is_b2b", "away_is_b2b",
                                     "home_rest", "away_rest",
                                     "home_season", "away_season"))]
    before = len(game_feats)
    game_feats = game_feats.dropna(subset=feature_cols)
    log.info("Dropped %d NaN rows → %d remaining", before - len(game_feats), len(game_feats))

    game_feats["game_date"] = game_feats["game_date"].astype(str)
    game_feats.to_csv(FEATURES_CSV, index=False)
    log.info("Saved → %s  (%d rows × %d cols)", FEATURES_CSV, *game_feats.shape)

    for season, grp in game_feats.groupby("season"):
        log.info("  %s: %d games, %.1f%% home-win", season, len(grp), grp["home_win"].mean()*100)


if __name__ == "__main__":
    main()
