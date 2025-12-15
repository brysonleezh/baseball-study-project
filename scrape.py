import time
import logging
import re
import statsapi
import pandas as pd

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -----------------------------
# Helpers
# -----------------------------
def safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_]+", "", name)
    return name

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

def get_regular_season_gamepks(team_id: int, season: int) -> list[int]:
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

# -----------------------------
# Core extraction: ONE ROW PER PA
# -----------------------------
def extract_target_player_plays_from_game(
    gamePk: int,
    target_batter_ids: set[int],
    season: int
) -> list[dict]:

    pbp = statsapi.get("game_playByPlay", {"gamePk": gamePk})
    rows = []

    for play in pbp.get("allPlays", []):
        about = play.get("about", {}) or {}
        matchup = play.get("matchup", {}) or {}
        batter = matchup.get("batter", {}) or {}
        pitcher = matchup.get("pitcher", {}) or {}

        batter_id = batter.get("id")
        if batter_id not in target_batter_ids:
            continue

        result = play.get("result", {}) or {}
        count = play.get("count", {}) or {}

        pitch_events = [e for e in play.get("playEvents", []) if e.get("isPitch")]
        last_pitch = pitch_events[-1] if pitch_events else {}

        details = last_pitch.get("details", {}) or {}
        pitch = last_pitch.get("pitchData", {}) or {}
        coords = pitch.get("coordinates", {}) or {}
        breaks = pitch.get("breaks", {}) or {}
        hit = last_pitch.get("hitData", {}) or {}

        runners = play.get("runners", []) or []
        runs_scored = sum(
            1 for r in runners
            if (r.get("details", {}) or {}).get("isScoringEvent") is True
        )

        rows.append({
            # ---- identifiers ----
            "season": season,
            "gamePk": gamePk,
            "inning": about.get("inning"),
            "halfInning": about.get("halfInning"),
            "atBatIndex": about.get("atBatIndex"),
            "startTime": about.get("startTime"),
            "endTime": about.get("endTime"),

            # ---- score context ----
            "homeScore": about.get("homeScore"),
            "awayScore": about.get("awayScore"),

            # ---- batter / pitcher ----
            "batter_id": batter.get("id"),
            "batter_name": batter.get("fullName"),
            "pitcher_id": pitcher.get("id"),
            "pitcher_name": pitcher.get("fullName"),
            "bat_side": (matchup.get("batSide") or {}).get("code"),
            "pitch_hand": (matchup.get("pitchHand") or {}).get("code"),

            # ---- PA result ----
            "event": result.get("event"),
            "eventType": result.get("eventType"),
            "description": result.get("description"),
            "rbi": result.get("rbi"),
            "runs_scored_on_play": runs_scored,

            # ---- count ----
            "balls": count.get("balls"),
            "strikes": count.get("strikes"),
            "outs": count.get("outs"),

            # ---- final pitch ----
            "pitch_number_last": last_pitch.get("pitchNumber"),
            "pitch_type_last": (details.get("type") or {}).get("code"),
            "start_speed_last": pitch.get("startSpeed"),
            "spin_rate_last": breaks.get("spinRate"),
            "spin_direction_last": breaks.get("spinDirection"),
            "plate_x_last": coords.get("pX"),
            "plate_z_last": coords.get("pZ"),
            "zone_last": pitch.get("zone"),

            # ---- batted-ball data ----
            "launch_speed": hit.get("launchSpeed"),
            "launch_angle": hit.get("launchAngle"),
            "hc_x": hit.get("hX"),
            "hc_y": hit.get("hY"),
        })

    return rows

# -----------------------------
# Main
# -----------------------------
def main():
    seasons = [2023, 2024, 2025]
    logging.info("===== PLAY (PA) EXTRACTION STARTED =====")

    marte_name = "Ketel Marte"
    carroll_name = "Corbin Carroll"

    marte_id = get_player_id(marte_name)
    carroll_id = get_player_id(carroll_name)
    target_ids = {marte_id, carroll_id}

    dbacks_id = get_team_id("Arizona Diamondbacks")

    all_rows = []

    for season in seasons:
        logging.info(f"----- SEASON {season} -----")
        gamepks = get_regular_season_gamepks(dbacks_id, season)

        for i, gpk in enumerate(gamepks, 1):
            try:
                rows = extract_target_player_plays_from_game(
                    gpk, target_ids, season
                )
                all_rows.extend(rows)

                marte_ct = sum(r["batter_id"] == marte_id for r in rows)
                carroll_ct = sum(r["batter_id"] == carroll_id for r in rows)

                logging.info(
                    f"[{season} {i}/{len(gamepks)}] gamePk={gpk} | "
                    f"rows={len(rows)} (Marte {marte_ct}, Carroll {carroll_ct})"
                )

            except Exception as e:
                logging.error(f"FAILED season={season} gamePk={gpk} | {e}")

            if i % 10 == 0:
                time.sleep(0.3)

    df = pd.DataFrame(all_rows)

    # -----------------------------
    # Save outputs
    # -----------------------------
    df_marte = df[df["batter_id"] == marte_id].reset_index(drop=True)
    df_carroll = df[df["batter_id"] == carroll_id].reset_index(drop=True)

    marte_file = f"{safe_filename(marte_name)}_plays_{min(seasons)}_{max(seasons)}.csv"
    carroll_file = f"{safe_filename(carroll_name)}_plays_{min(seasons)}_{max(seasons)}.csv"

    df_marte.to_csv(marte_file, index=False)
    df_carroll.to_csv(carroll_file, index=False)

    logging.info(f"Saved Marte:   {len(df_marte)} rows → {marte_file}")
    logging.info(f"Saved Carroll: {len(df_carroll)} rows → {carroll_file}")
    logging.info("===== EXTRACTION COMPLETE =====")

    return df_marte, df_carroll


if __name__ == "__main__":
    df_marte, df_carroll = main()
