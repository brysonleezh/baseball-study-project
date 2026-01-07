#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import statsapi
import pyodbc


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# -----------------------------
# Config (hidden file)
# -----------------------------
DEFAULT_DB_CONFIG_PATH = ".db_config.json"


def load_db_config(path: str = DEFAULT_DB_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load SQL Server connection config from a hidden json file.
    Fallback to env vars if file missing.

    .db_config.json example:
    {
      "driver": "ODBC Driver 18 for SQL Server",
      "server": "localhost,1433",
      "database": "baseball",
      "username": "sa",
      "password": "YourStrong!Passw0rd",
      "encrypt": "yes",
      "trust_server_certificate": "yes"
    }
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg

    # env fallback
    cfg = {
        "driver": os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server"),
        "server": os.getenv("MSSQL_SERVER", ""),
        "database": os.getenv("MSSQL_DATABASE", ""),
        "username": os.getenv("MSSQL_USERNAME", ""),
        "password": os.getenv("MSSQL_PASSWORD", ""),
        "encrypt": os.getenv("MSSQL_ENCRYPT", "yes"),
        "trust_server_certificate": os.getenv("MSSQL_TRUST_SERVER_CERTIFICATE", "yes"),
    }
    return cfg


def build_conn_str(cfg: Dict[str, Any]) -> str:
    missing = [k for k in ["server", "database", "username", "password"] if not cfg.get(k)]
    if missing:
        raise ValueError(
            f"Missing DB config fields: {missing}. "
            f"Create {DEFAULT_DB_CONFIG_PATH} or set env vars."
        )

    driver = cfg.get("driver", "ODBC Driver 18 for SQL Server")
    encrypt = cfg.get("encrypt", "yes")
    trust = cfg.get("trust_server_certificate", "yes")

    # NOTE: If you use Windows auth, you'd change this format.
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={cfg['server']};"
        f"DATABASE={cfg['database']};"
        f"UID={cfg['username']};"
        f"PWD={cfg['password']};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust};"
    )
    return conn_str


def get_sqlserver_conn(config_path: str = DEFAULT_DB_CONFIG_PATH) -> pyodbc.Connection:
    cfg = load_db_config(config_path)
    conn_str = build_conn_str(cfg)
    # autocommit=False so we control commits
    return pyodbc.connect(conn_str, autocommit=False)


# -----------------------------
# SQL DDL
# -----------------------------
TABLE_NAME = "dbo.statcast_plays"

CREATE_TABLE_SQL = f"""
IF OBJECT_ID('{TABLE_NAME}', 'U') IS NULL
BEGIN
    CREATE TABLE {TABLE_NAME} (
        play_id BIGINT IDENTITY(1,1) PRIMARY KEY,

        season INT NULL,
        gamePk INT NULL,
        inning INT NULL,
        halfInning VARCHAR(10) NULL,
        atBatIndex INT NULL,

        startTime DATETIME2 NULL,
        endTime DATETIME2 NULL,

        homeScore INT NULL,
        awayScore INT NULL,

        batter_id INT NULL,
        batter_name NVARCHAR(100) NULL,
        pitcher_id INT NULL,
        pitcher_name NVARCHAR(100) NULL,

        bat_side CHAR(1) NULL,
        pitch_hand CHAR(1) NULL,

        event NVARCHAR(50) NULL,
        eventType NVARCHAR(50) NULL,
        description NVARCHAR(500) NULL,

        rbi INT NULL,
        runs_scored_on_play INT NULL,
        balls INT NULL,
        strikes INT NULL,
        outs INT NULL,

        pitch_number_last INT NULL,
        pitch_type_last NVARCHAR(10) NULL,
        start_speed_last FLOAT NULL,
        spin_rate_last FLOAT NULL,
        spin_direction_last FLOAT NULL,
        plate_x_last FLOAT NULL,
        plate_z_last FLOAT NULL,
        zone_last INT NULL,

        launch_speed FLOAT NULL,
        launch_angle FLOAT NULL,
        hc_x FLOAT NULL,
        hc_y FLOAT NULL,

        inserted_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
    );
END;
"""

# Create UNIQUE index if not exists
CREATE_UNIQUE_INDEX_SQL = f"""
IF NOT EXISTS (
    SELECT 1
    FROM sys.indexes
    WHERE name = 'UX_statcast_nodup'
      AND object_id = OBJECT_ID('{TABLE_NAME}')
)
BEGIN
    CREATE UNIQUE INDEX UX_statcast_nodup
    ON {TABLE_NAME} (gamePk, atBatIndex, pitch_number_last);
END;
"""


# IMPORTANT: columns we insert (no play_id, no inserted_at)
INSERT_COLS: List[str] = [
    "season", "gamePk", "inning", "halfInning", "atBatIndex",
    "startTime", "endTime",
    "homeScore", "awayScore",
    "batter_id", "batter_name", "pitcher_id", "pitcher_name",
    "bat_side", "pitch_hand",
    "event", "eventType", "description",
    "rbi", "runs_scored_on_play", "balls", "strikes", "outs",
    "pitch_number_last", "pitch_type_last", "start_speed_last",
    "spin_rate_last", "spin_direction_last",
    "plate_x_last", "plate_z_last", "zone_last",
    "launch_speed", "launch_angle", "hc_x", "hc_y"
]

KEY_COLS: List[str] = ["gamePk", "atBatIndex", "pitch_number_last"]


def ensure_schema(conn: pyodbc.Connection) -> None:
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    cur.execute(CREATE_UNIQUE_INDEX_SQL)
    conn.commit()


# -----------------------------
# Helpers
# -----------------------------
def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "", name)
    return name


def parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """
    StatsAPI often returns ISO like '2023-03-31T02:26:17.244Z'
    SQL DATETIME2 has no tz; we store UTC as naive datetime.
    """
    if not s:
        return None
    try:
        # normalize Z -> +00:00 for fromisoformat
        ss = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ss)
        # convert to UTC and drop tzinfo
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


def get_player_id(full_name: str) -> int:
    hits = statsapi.lookup_player(full_name)
    if not hits:
        raise ValueError(f"Player not found: {full_name}")
    exact = [h for h in hits if h.get("fullName", "").lower() == full_name.lower()]
    pick = exact[0] if exact else hits[0]
    pid = int(pick["id"])
    logging.info(f"Resolved player ID: {full_name} → {pid}")
    return pid


def get_team_id(team_name: str) -> int:
    hits = statsapi.lookup_team(team_name)
    if not hits:
        raise ValueError(f"Team not found: {team_name}")
    tid = int(hits[0]["id"])
    logging.info(f"Resolved team ID: {team_name} → {tid}")
    return tid


def get_regular_season_gamepks(team_id: int, season: int) -> List[int]:
    logging.info(f"Fetching {season} regular-season schedule for teamId={team_id}...")
    sch = statsapi.get("schedule", {
        "sportId": 1,
        "season": season,
        "teamId": team_id,
        "gameTypes": "R",
    })
    gamepks = [
        int(g["gamePk"])
        for d in sch.get("dates", [])
        for g in d.get("games", [])
    ]
    logging.info(f"Found {len(gamepks)} regular-season games in {season}")
    return gamepks

def get_home_away_team_ids(gamePk: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Robustly get (home_id, away_id). Some statsapi payloads may not include gameData.teams ids.
    """
    # 1) try from playByPlay response
    pbp = statsapi.get("game_playByPlay", {"gamePk": gamePk})
    gd = pbp.get("gameData") or {}
    teams = gd.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_id = home.get("id")
    away_id = away.get("id")

    # 2) fallback: use 'game' endpoint (often contains gameData reliably)
    if home_id is None or away_id is None:
        try:
            g = statsapi.get("game", {"gamePk": gamePk})
            gd2 = g.get("gameData") or {}
            teams2 = gd2.get("teams") or {}
            home2 = teams2.get("home") or {}
            away2 = teams2.get("away") or {}
            home_id = home_id if home_id is not None else home2.get("id")
            away_id = away_id if away_id is not None else away2.get("id")
        except Exception:
            pass

    # normalize to int
    try:
        home_id = int(home_id) if home_id is not None else None
    except Exception:
        home_id = None
    try:
        away_id = int(away_id) if away_id is not None else None
    except Exception:
        away_id = None

    return home_id, away_id


# -----------------------------
# Core extraction: ONE ROW PER PA
# -----------------------------
def extract_team_plays_from_game(
    gamePk: int,
    season: int,
    team_id: int,
    target_ids: Optional[set] = None  # None => no player filter
) -> List[Dict[str, Any]]:

    pbp = statsapi.get("game_playByPlay", {"gamePk": gamePk})

    home_id, away_id = get_home_away_team_ids(gamePk)
    if home_id is None or away_id is None:
        # If we can't identify teams, don't hard-filter (better to return something than 0 forever)
        # But we still can choose to return empty.
        logging.warning(f"Could not resolve home/away IDs for gamePk={gamePk}.")
        return []

    rows: List[Dict[str, Any]] = []

    for play in pbp.get("allPlays", []) or []:
        
        
        about = play.get("about", {}) or {}
        matchup = play.get("matchup", {}) or {}

        half = (about.get("halfInning") or "").lower()
        if half == "top":
            batting_team_id = away_id
        elif half == "bottom":
            batting_team_id = home_id
        else:
            continue

        # keep only target team batters
        if int(batting_team_id) != int(team_id):
            continue

        batter = matchup.get("batter", {}) or {}
        batter_id = batter.get("id")

        # optional: player filter
        if target_ids is not None:
            try:
                if batter_id is None or int(batter_id) not in target_ids:
                    continue
            except Exception:
                continue

        pitcher = matchup.get("pitcher", {}) or {}

        result = play.get("result", {}) or {}
        count = play.get("count", {}) or {}

        pitch_events = [e for e in (play.get("playEvents") or []) if e.get("isPitch")]
        last_pitch = pitch_events[-1] if pitch_events else {}

        details = last_pitch.get("details", {}) or {}
        pitch = last_pitch.get("pitchData", {}) or {}
        coords = pitch.get("coordinates", {}) or {}
        breaks = pitch.get("breaks", {}) or {}
        hit = last_pitch.get("hitData", {}) or {}

        runners = play.get("runners", []) or []
        runs_scored = sum(
            1 for r in runners
            if ((r.get("details") or {}) or {}).get("isScoringEvent") is True
        )

        rows.append({
            "season": season,
            "gamePk": gamePk,
            "inning": about.get("inning"),
            "halfInning": about.get("halfInning"),
            "atBatIndex": about.get("atBatIndex"),
            "startTime": parse_iso_utc(about.get("startTime")),
            "endTime": parse_iso_utc(about.get("endTime")),
            "homeScore": about.get("homeScore"),
            "awayScore": about.get("awayScore"),

            "batter_id": batter.get("id"),
            "batter_name": batter.get("fullName"),
            "pitcher_id": pitcher.get("id"),
            "pitcher_name": pitcher.get("fullName"),
            "bat_side": (matchup.get("batSide") or {}).get("code"),
            "pitch_hand": (matchup.get("pitchHand") or {}).get("code"),

            "event": result.get("event"),
            "eventType": result.get("eventType"),
            "description": result.get("description"),
            "rbi": result.get("rbi"),
            "runs_scored_on_play": runs_scored,

            "balls": count.get("balls"),
            "strikes": count.get("strikes"),
            "outs": count.get("outs"),

            "pitch_number_last": last_pitch.get("pitchNumber"),
            "pitch_type_last": ((details.get("type") or {}) or {}).get("code"),
            "start_speed_last": pitch.get("startSpeed"),
            "spin_rate_last": breaks.get("spinRate"),
            "spin_direction_last": breaks.get("spinDirection"),
            "plate_x_last": coords.get("pX"),
            "plate_z_last": coords.get("pZ"),
            "zone_last": pitch.get("zone"),

            "launch_speed": hit.get("launchSpeed"),
            "launch_angle": hit.get("launchAngle"),
            "hc_x": hit.get("hX"),
            "hc_y": hit.get("hY"),
        })

    return rows




# -----------------------------
# DB ingest (dedup)
# -----------------------------
def build_insert_sql() -> str:
    """
    INSERT ... SELECT ... WHERE NOT EXISTS ...
    Total params = len(INSERT_COLS) + len(KEY_COLS) = 36 + 3 = 39
    """
    col_list = ",".join(INSERT_COLS)
    placeholders = ",".join(["?"] * len(INSERT_COLS))

    # key placeholders appear AGAIN in NOT EXISTS
    where_clause = " AND ".join([f"{k} = ?" for k in KEY_COLS])

    sql = f"""
    INSERT INTO {TABLE_NAME} ({col_list})
    SELECT {placeholders}
    WHERE NOT EXISTS (
        SELECT 1 FROM {TABLE_NAME}
        WHERE {where_clause}
    );
    """
    return sql


INSERT_SQL = build_insert_sql()


def row_to_params(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Params order must match:
    [INSERT_COLS...] + [KEY_COLS...]
    """
    base = [row.get(c) for c in INSERT_COLS]
    keys = [row.get(k) for k in KEY_COLS]
    return tuple(base + keys)


def ingest_rows(conn: pyodbc.Connection, rows: List[Dict[str, Any]], batch_size: int = 500) -> int:
    if not rows:
        return 0

    cur = conn.cursor()
    cur.fast_executemany = True

    params_list = [row_to_params(r) for r in rows]

    # Safety check: make sure we never mismatch placeholders
    expected = INSERT_SQL.count("?")
    for p in params_list[:3]:
        if len(p) != expected:
            raise ValueError(f"Param mismatch: expected {expected}, got {len(p)}")

    inserted = 0
    # We can't directly know how many inserted with NOT EXISTS using executemany reliably,
    # but we can estimate by counting rowcount per batch in many cases.
    for i in range(0, len(params_list), batch_size):
        chunk = params_list[i:i + batch_size]
        cur.executemany(INSERT_SQL, chunk)
        # rowcount behavior can vary; still useful as an approximation
        if cur.rowcount and cur.rowcount > 0:
            inserted += cur.rowcount

    conn.commit()
    return inserted


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape MLB play-by-play (one row per PA) via statsapi and ingest into SQL Server."
    )
    p.add_argument("--team", required=True, help='Team name, e.g. "Arizona Diamondbacks"')
    p.add_argument("--seasons", nargs="+", required=True, type=int, help="Seasons, e.g. 2023 2024 2025")
    p.add_argument(
        "--players",
        nargs="*",
        default=None,
        help='Optional player full names. If omitted, scrape ALL batters on the team.'
    )
    p.add_argument("--sleep_every", type=int, default=10, help="Sleep after every N games (rate-limit friendly)")
    p.add_argument("--sleep_sec", type=float, default=0.3, help="Sleep seconds")
    p.add_argument("--db_config", default=DEFAULT_DB_CONFIG_PATH, help="Path to .db_config.json")
    p.add_argument("--save_csv", action="store_true", help="Also save a local CSV snapshot")
    p.add_argument("--csv_dir", default=".", help="Where to save csv if --save_csv")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    logging.info("===== STATAPI -> SQL SERVER INGEST START =====")
    logging.info(f"Team: {args.team}")
    logging.info(f"Seasons: {args.seasons}")
    logging.info(f"Players: {args.players if args.players else '[ALL TEAM BATTERS]'}")

    # Resolve team
    team_id = get_team_id(args.team)

    # Resolve player filter (optional)
    if args.players:
        player_ids: Dict[str, int] = {name: get_player_id(name) for name in args.players}
        target_ids = set(player_ids.values())
    else:
        player_ids = {}
        target_ids = None  # None means "no filter"

    # Connect DB + ensure schema
    conn = get_sqlserver_conn(args.db_config)
    ensure_schema(conn)

    all_rows: List[Dict[str, Any]] = []

    # Scrape
    for season in args.seasons:
        gamepks = get_regular_season_gamepks(team_id, season)

        for i, gpk in enumerate(gamepks, 1):
            try:
                # IMPORTANT: pass target_ids (can be None)
                rows = extract_team_plays_from_game(gpk, season, team_id, target_ids)
#                 print(len(rows), {r["halfInning"] for r in rows}, rows[0]["batter_name"] if rows else None)


                inserted = ingest_rows(conn, rows, batch_size=500) if rows else 0
                unique_batters = len({r["batter_id"] for r in rows if r.get("batter_id") is not None})


                all_rows.extend(rows)

                # Logging
                if args.players:
                    per_player_ct = {
                        name: sum(r["batter_id"] == pid for r in rows)
                        for name, pid in player_ids.items()
                    }
                    extra_log = f"{per_player_ct}"
                else:
                    unique_batters = len({r["batter_id"] for r in rows if r.get("batter_id") is not None})
                    extra_log = f"unique_batters={unique_batters}"

                logging.info(
                    f"[{season} {i}/{len(gamepks)}] gamePk={gpk} | "
                    f"scraped={len(rows)} inserted~={inserted} | {extra_log}"
                )

            except Exception as e:
                logging.error(f"FAILED season={season} gamePk={gpk} | {e}")

            if args.sleep_every > 0 and i % args.sleep_every == 0:
                time.sleep(args.sleep_sec)

    logging.info("===== INGEST COMPLETE =====")

    # Optional CSV snapshot
    if args.save_csv:
        df = pd.DataFrame(all_rows)
        os.makedirs(args.csv_dir, exist_ok=True)

        players_tag = safe_filename("_".join(args.players)) if args.players else "ALL"
        out = os.path.join(
            args.csv_dir,
            f"{safe_filename(args.team)}_{min(args.seasons)}_{max(args.seasons)}_{players_tag}.csv"
        )

        df.to_csv(out, index=False)
        logging.info(f"Saved CSV snapshot: {out} (rows={len(df)})")

    conn.close()


if __name__ == "__main__":
    main()
