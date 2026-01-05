#!/usr/bin/env python3
"""
Build MLB player profiles from a list of names using MLB StatsAPI.

Input:
  - CSV with column: batter_name
Output:
  - player_profiles.csv with columns:
      batter_name, mlbam_id, matched_name, match_score,
      team, position, bats, throws,
      height, weight, birth_date, age,
      bio, headshot_url

Usage:
  python build_player_profiles.py --input batter_names.csv --output player_profiles.csv

Notes:
  - This script matches names via /api/v1/people/search and then hydrates player info.
  - Name matching is fuzzy to handle accent/encoding issues.
"""

import argparse
import csv
import datetime as dt
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from difflib import SequenceMatcher


BASE = "https://statsapi.mlb.com/api/v1"


def normalize_name(s: str) -> str:
    """Normalize names for matching: lower, remove extra spaces/punct, strip weird chars."""
    if s is None:
        return ""
    s = s.strip()

    # Fix common mojibake-like artifacts (best effort)
    # If your source CSV has these, consider re-saving it as UTF-8 as well.
    s = s.replace("√°", "á").replace("√∫", "ó").replace("√≠", "í")

    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # remove punctuation except spaces and letters
    s = re.sub(r"[^a-z\s\.\-']", "", s)
    s = s.replace(".", "")
    return s.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def compute_age(birth_date: Optional[str]) -> Optional[int]:
    if not birth_date:
        return None
    try:
        y, m, d = map(int, birth_date.split("-"))
        born = dt.date(y, m, d)
        today = dt.date.today()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return age
    except Exception:
        return None


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class MatchResult:
    mlbam_id: Optional[int]
    matched_name: Optional[str]
    score: float
    raw_candidate: Optional[Dict[str, Any]]


class MLBStatsAPI:
    def __init__(self, session: Optional[requests.Session] = None, sleep_s: float = 0.15):
        self.sess = session or requests.Session()
        self.sleep_s = sleep_s

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{BASE}{endpoint}"
        r = self.sess.get(url, params=params, timeout=30)
        r.raise_for_status()
        time.sleep(self.sleep_s)  # be polite
        return r.json()

    def search_people(self, name: str, sport_id: int = 1) -> List[Dict[str, Any]]:
        # /people/search is the most practical for name->id resolution
        data = self._get("/people/search", params={"names": name, "sportId": sport_id})
        # StatsAPI returns "people" or "search_player_all" depending on endpoint behavior;
        # this handles the common one.
        return data.get("people", []) or data.get("search_player_all", []) or []

    def get_person(self, person_id: int) -> Optional[Dict[str, Any]]:
        # hydrate=currentTeam is helpful for team
        data = self._get(f"/people/{person_id}", params={"hydrate": "currentTeam"})
        people = data.get("people", [])
        return people[0] if people else None


def pick_best_match(query_name: str, candidates: List[Dict[str, Any]]) -> MatchResult:
    if not candidates:
        return MatchResult(None, None, 0.0, None)

    # Score by fullName similarity + small bonuses if exact last name match, etc.
    q_norm = normalize_name(query_name)

    best = None
    best_score = -1.0

    for c in candidates:
        full = c.get("fullName") or c.get("nameFirstLast") or ""
        score = similarity(query_name, full)

        # Bonus: exact normalized match
        if normalize_name(full) == q_norm:
            score += 0.15

        # Bonus: same last name token
        q_last = q_norm.split(" ")[-1] if q_norm else ""
        c_last = normalize_name(full).split(" ")[-1] if full else ""
        if q_last and c_last and q_last == c_last:
            score += 0.05

        if score > best_score:
            best_score = score
            best = c

    mlbam_id = best.get("id") if best else None
    matched_name = best.get("fullName") if best else None
    return MatchResult(mlbam_id, matched_name, float(best_score), best)


def build_headshot_url(mlbam_id: int) -> str:
    """
    MLB headshot URLs are hosted on MLB static/CDN.
    This pattern is commonly used for silo headshots.
    If MLB changes the URL, you can switch to saving local files instead.
    """
    return f"https://img.mlbstatic.com/mlb-photos/image/upload/w_213,q_auto:best/v1/people/{mlbam_id}/headshot/silo/current"


def make_bio(p: Dict[str, Any]) -> str:
    """
    StatsAPI doesn't always provide a rich 'bio' field.
    We'll synthesize a short one from known attributes.
    """
    full = p.get("fullName", "Unknown")
    pos = safe_get(p, ["primaryPosition", "name"], "")
    bats = safe_get(p, ["batSide", "description"], "")
    throws = safe_get(p, ["pitchHand", "description"], "")
    team = safe_get(p, ["currentTeam", "name"], "")

    parts = [full]
    if pos:
        parts.append(pos)
    if bats or throws:
        parts.append(f"Bats {bats or '—'} / Throws {throws or '—'}")
    if team:
        parts.append(f"{team}")
    return " • ".join(parts)


def load_names(path: str) -> List[str]:
    names = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "batter_name" not in reader.fieldnames:
            raise ValueError(f"Input CSV must have a 'batter_name' column. Got: {reader.fieldnames}")
        for row in reader:
            n = (row.get("batter_name") or "").strip()
            if n:
                names.append(n)
    # de-dup while preserving order
    seen = set()
    out = []
    for n in names:
        key = normalize_name(n)
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV file with column batter_name")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--min_score", type=float, default=0.72, help="Min fuzzy match score to accept")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between API calls (seconds)")
    args = ap.parse_args()

    names = load_names(args.input)
    api = MLBStatsAPI(sleep_s=args.sleep)

    rows: List[Dict[str, Any]] = []

    for i, name in enumerate(names, 1):
        try:
            candidates = api.search_people(name, sport_id=1)
            match = pick_best_match(name, candidates)

            if match.mlbam_id is None or match.score < args.min_score:
                rows.append({
                    "batter_name": name,
                    "mlbam_id": "",
                    "matched_name": match.matched_name or "",
                    "match_score": f"{match.score:.3f}",
                    "team": "",
                    "position": "",
                    "bats": "",
                    "throws": "",
                    "height": "",
                    "weight": "",
                    "birth_date": "",
                    "age": "",
                    "bio": "",
                    "headshot_url": "",
                })
                print(f"[{i}/{len(names)}] NO MATCH (score={match.score:.3f}): {name}")
                continue

            person = api.get_person(int(match.mlbam_id))
            if not person:
                raise RuntimeError("Could not hydrate person info")

            team = safe_get(person, ["currentTeam", "name"], "")
            position = safe_get(person, ["primaryPosition", "name"], "")
            bats = safe_get(person, ["batSide", "code"], "")  # L/R/S
            throws = safe_get(person, ["pitchHand", "code"], "")  # L/R
            height = person.get("height", "")
            weight = person.get("weight", "")
            birth_date = person.get("birthDate", "")
            age = compute_age(birth_date)

            mlbam_id = int(person.get("id"))
            headshot_url = build_headshot_url(mlbam_id)
            bio = make_bio(person)

            rows.append({
                "batter_name": name,
                "mlbam_id": mlbam_id,
                "matched_name": person.get("fullName", match.matched_name or ""),
                "match_score": f"{match.score:.3f}",
                "team": team,
                "position": position,
                "bats": bats,
                "throws": throws,
                "height": height,
                "weight": weight,
                "birth_date": birth_date,
                "age": age if age is not None else "",
                "bio": bio,
                "headshot_url": headshot_url,
            })

            print(f"[{i}/{len(names)}] OK (score={match.score:.3f}) {name} -> {mlbam_id} ({team})")

        except Exception as e:
            rows.append({
                "batter_name": name,
                "mlbam_id": "",
                "matched_name": "",
                "match_score": "",
                "team": "",
                "position": "",
                "bats": "",
                "throws": "",
                "height": "",
                "weight": "",
                "birth_date": "",
                "age": "",
                "bio": "",
                "headshot_url": "",
                "error": str(e),
            })
            print(f"[{i}/{len(names)}] ERROR: {name} -> {e}")

    # Ensure stable column order
    cols = [
        "batter_name", "mlbam_id", "matched_name", "match_score",
        "team", "position", "bats", "throws",
        "height", "weight", "birth_date", "age",
        "bio", "headshot_url"
    ]
    # If any row contains "error", include it
    if any("error" in r for r in rows):
        cols.append("error")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved: {args.output}  (rows={len(rows)})")


if __name__ == "__main__":
    main()
