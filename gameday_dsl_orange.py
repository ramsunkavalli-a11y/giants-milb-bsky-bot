import argparse
import html
import json
import os
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from atproto import Client, models

from bot import STATE_PATH, SLEEP_BETWEEN_POSTS_SEC, make_session

# Local testing quick-start:
#   DRY_RUN=1 python gameday_dsl_orange.py
#   DRY_RUN=1 OVERRIDE_DATE=2025-07-18 python gameday_dsl_orange.py
#   DRY_RUN=1 OVERRIDE_GAMEPK=811804 FORCE_REPOST=1 python gameday_dsl_orange.py

API_BASE = "https://statsapi.mlb.com"
DSL_ORANGE_TEAM_ID = 615
SPORT_ID = 16
MAX_POST_CHARS = 300
PROSPECTS_PATH = Path("prospects.json")
PROSPECTS_STALE_DAYS = 45
PLAYER_CACHE_PATH = Path("player_cache.json")
TANGO_WE_PATH = Path("data/tango_we.json")


def _normalize_name(name: str) -> str:
    base = unicodedata.normalize("NFKD", (name or "").strip().lower())
    return " ".join("".join(ch for ch in base if not unicodedata.combining(ch)).split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ip_to_outs(ip: str) -> int:
    parts = str(ip or "0.0").split(".")
    whole = _safe_int(parts[0])
    frac = min(max(_safe_int(parts[1]) if len(parts) > 1 else 0, 0), 2)
    return whole * 3 + frac


def _load_json(path: Path, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_state() -> Dict[str, Any]:
    state = _load_json(Path(STATE_PATH), {"bootstrapped": False, "seen_transaction_ids": [], "last_run_iso": None, "posted_games": {}, "recaps": {}})
    state.setdefault("posted_games", {})
    state.setdefault("recaps", {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    _save_json(Path(STATE_PATH), state)


def bsky_login() -> Client:
    client = Client()
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"])
    return client


def fetch_schedule_games(
    session: requests.Session,
    target_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"sportId": SPORT_ID, "teamId": DSL_ORANGE_TEAM_ID}
    if target_date:
        params["date"] = target_date
    else:
        params["startDate"] = start_date or (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()
        params["endDate"] = end_date or datetime.now(timezone.utc).date().isoformat()
    try:
        res = session.get(f"{API_BASE}/api/v1/schedule", params=params, timeout=30)
        res.raise_for_status()
        dates = (res.json() or {}).get("dates", [])
    except requests.RequestException as exc:
        print(f"WARNING: schedule fetch failed: {exc}")
        return []

    games: List[Dict[str, Any]] = []
    for d in dates:
        games.extend(d.get("games", []))
    return games


def fetch_game_feed(session: requests.Session, game_pk: int) -> Dict[str, Any]:
    try:
        res = session.get(f"{API_BASE}/api/v1.1/game/{game_pk}/feed/live", params={"language": "en"}, timeout=30)
        res.raise_for_status()
        return res.json() or {}
    except requests.RequestException as exc:
        print(f"WARNING: game feed fetch failed for {game_pk}: {exc}")
        return {}


def classify_terminal_status(feed: Dict[str, Any]) -> Optional[str]:
    status = ((feed.get("gameData") or {}).get("status") or {})
    detailed = (status.get("detailedState") or "").lower()
    coded = (status.get("codedGameState") or "").upper()
    abstract = (status.get("abstractGameState") or "").lower()
    if "suspend" in detailed:
        return "Suspended"
    if detailed in {"final", "game over"} or coded == "F" or abstract == "final":
        return "Final"
    return None


def _player_season_slash(player: Dict[str, Any]) -> Optional[str]:
    season_bat = ((player.get("seasonStats") or {}).get("batting") or {})
    avg, obp, slg = season_bat.get("avg"), season_bat.get("obp"), season_bat.get("slg")
    if avg and obp and slg:
        return f"{avg}/{obp}/{slg}"
    return None


def _load_prospects() -> Dict[str, Any]:
    return _load_json(PROSPECTS_PATH, {"updated": "", "prospects": []})


def _prospect_maps() -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    name_map: Dict[str, Dict[str, Any]] = {}
    id_map: Dict[int, Dict[str, Any]] = {}
    for p in _load_prospects().get("prospects", []):
        nm = _normalize_name(p.get("name", ""))
        if nm:
            name_map[nm] = p
        pid = p.get("personId")
        if pid is not None:
            id_map[_safe_int(pid)] = p
    return name_map, id_map


def maybe_warn_stale_prospects() -> None:
    if not PROSPECTS_PATH.exists():
        print("WARNING: prospects.json not found")
        return
    mtime = datetime.fromtimestamp(PROSPECTS_PATH.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    if age_days > PROSPECTS_STALE_DAYS:
        print(f"WARNING: prospects.json is {age_days} days old; consider refreshing rankings.")


def _load_player_cache() -> Dict[str, Any]:
    return _load_json(PLAYER_CACHE_PATH, {"players": {}})


def _save_player_cache(cache: Dict[str, Any]) -> None:
    _save_json(PLAYER_CACHE_PATH, cache)


def _get_person_details(session: requests.Session, person_id: int, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = str(person_id)
    players = cache.setdefault("players", {})
    if key in players:
        return players[key]
    detail = {"pitchHand": "", "primaryPos": ""}
    try:
        res = session.get(f"{API_BASE}/api/v1/people/{person_id}", timeout=30)
        res.raise_for_status()
        person = ((res.json() or {}).get("people") or [{}])[0]
        detail = {
            "pitchHand": ((person.get("pitchHand") or {}).get("code") or "").upper(),
            "primaryPos": ((person.get("primaryPosition") or {}).get("abbreviation") or ""),
        }
    except requests.RequestException:
        pass
    players[key] = detail
    return detail


def _derive_pitch_metrics(feed: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    metrics: Dict[int, Dict[str, int]] = {}
    plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
    for play in plays:
        pid = _safe_int(((play.get("matchup") or {}).get("pitcher") or {}).get("id"), -1)
        if pid <= 0:
            continue
        pm = metrics.setdefault(pid, {"swstr": 0, "gb": 0})
        for pe in play.get("playEvents") or []:
            if not pe.get("isPitch"):
                continue
            desc = ((pe.get("details") or {}).get("description") or "").lower()
            code = ((pe.get("details") or {}).get("code") or "").upper()
            if "swinging strike" in desc or code in {"S", "W", "Q", "T"}:
                pm["swstr"] += 1
            trajectory = ((pe.get("hitData") or {}).get("trajectory") or "").lower()
            if "ground" in trajectory:
                pm["gb"] += 1
    return metrics


def _primary_hitter_position(all_positions: List[Dict[str, Any]]) -> str:
    non_p = [p.get("abbreviation", "") for p in all_positions if p.get("abbreviation") not in {"P", ""}]
    return (non_p[0] if non_p else (all_positions[0].get("abbreviation", "") if all_positions else "")) or "-"


def _sub_position(all_positions: List[Dict[str, Any]]) -> str:
    if not all_positions:
        return "-"
    for p in reversed(all_positions):
        abbr = p.get("abbreviation") or ""
        if abbr and abbr != "P":
            return abbr
    return all_positions[-1].get("abbreviation") or "-"


def extract_player_lines(feed: Dict[str, Any], session: requests.Session) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    game_data = feed.get("gameData") or {}
    live_data = feed.get("liveData") or {}
    boxscore = live_data.get("boxscore") or {}
    teams = boxscore.get("teams") or {}

    orange_is_home = (((game_data.get("teams") or {}).get("home") or {}).get("id") == DSL_ORANGE_TEAM_ID)
    side_key = "home" if orange_is_home else "away"
    orange_team = teams.get(side_key) or {}
    player_map = orange_team.get("players") or {}

    batters_set = {_safe_int(x, -1) for x in (orange_team.get("batters") or [])}
    pitchers_order = [_safe_int(x, -1) for x in (orange_team.get("pitchers") or [])]
    pitchers_set = set(pitchers_order)
    pitch_metrics = _derive_pitch_metrics(feed)
    cache = _load_player_cache()

    slot_groups: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(1, 10)}
    unknown_subs: List[Dict[str, Any]] = []
    pitchers: List[Dict[str, Any]] = []

    for pdata in player_map.values():
        person = pdata.get("person") or {}
        pid = _safe_int(person.get("id"), -1)
        name = person.get("fullName", "Unknown")
        all_positions = pdata.get("allPositions") or []
        batting = ((pdata.get("stats") or {}).get("batting") or {})
        pitching = ((pdata.get("stats") or {}).get("pitching") or {})
        bo_raw = str(pdata.get("battingOrder") or "").strip()
        game_status = pdata.get("gameStatus") or {}

        appeared_hitter = bool(bo_raw) or bool(all_positions) or (game_status.get("isOnBench") is False) or pid in batters_set
        if appeared_hitter:
            pa = _safe_int(batting.get("plateAppearances"), _safe_int(batting.get("atBats")) + _safe_int(batting.get("baseOnBalls")))
            doubles = _safe_int(batting.get("doubles"))
            triples = _safe_int(batting.get("triples"))
            hr = _safe_int(batting.get("homeRuns"))
            hits = _safe_int(batting.get("hits"))
            tb = _safe_int(batting.get("totalBases"), hits + doubles + 2 * triples + 3 * hr)
            hitter_row = {
                "id": pid,
                "name": name,
                "pos": _primary_hitter_position(all_positions),
                "pos_sub": _sub_position(all_positions),
                "pa": pa,
                "ab": _safe_int(batting.get("atBats")),
                "r": _safe_int(batting.get("runs")),
                "h": hits,
                "bb": _safe_int(batting.get("baseOnBalls")),
                "k": _safe_int(batting.get("strikeOuts")),
                "2b": doubles,
                "3b": triples,
                "hr": hr,
                "sb": _safe_int(batting.get("stolenBases")),
                "tb": max(0, tb),
                "season_slash": _player_season_slash(pdata),
                "batting_order": bo_raw,
                "is_sub": False,
                "indent": False,
            }
            if bo_raw.isdigit():
                bo = int(bo_raw)
                slot = bo // 100
                seq = bo % 100
                hitter_row["seq"] = seq
                hitter_row["is_sub"] = seq > 0
                if 1 <= slot <= 9:
                    slot_groups[slot].append(hitter_row)
                else:
                    unknown_subs.append(hitter_row)
            else:
                unknown_subs.append(hitter_row)

        appeared_pitcher = pid in pitchers_set or bool(pitching) or any(k in pitching for k in ("inningsPitched", "numberOfPitches", "strikeOuts"))
        if appeared_pitcher:
            pmeta = _get_person_details(session, pid, cache)
            hand_code = pmeta.get("pitchHand") or ""
            hand = "RHP" if hand_code == "R" else "LHP" if hand_code == "L" else "P"
            ip = str(pitching.get("inningsPitched") or "0.0")
            outs = _ip_to_outs(ip)
            hits_allowed = _safe_int(pitching.get("hits"))
            walks = _safe_int(pitching.get("baseOnBalls"))
            ip_val = max(outs / 3.0, 0.001)
            whip = round((hits_allowed + walks) / ip_val, 2)
            pm = pitch_metrics.get(pid, {})
            pitchers.append(
                {
                    "id": pid,
                    "name": name,
                    "hand": hand,
                    "ip": ip,
                    "ip_outs": outs,
                    "h": hits_allowed,
                    "r": _safe_int(pitching.get("runs")),
                    "er": _safe_int(pitching.get("earnedRuns")),
                    "bb": walks,
                    "k": _safe_int(pitching.get("strikeOuts")),
                    "bf": _safe_int(pitching.get("battersFaced")),
                    "whip": f"{whip:.2f}",
                    "swstr": pm.get("swstr") or "",
                    "gb": pm.get("gb") or "",
                }
            )

    hitters: List[Dict[str, Any]] = []
    for slot in range(1, 10):
        rows = sorted(slot_groups[slot], key=lambda x: (x.get("seq", 999), x["name"]))
        for i, row in enumerate(rows):
            row["slot"] = slot
            row["indent"] = i > 0
            if row["indent"]:
                row["pos"] = row.get("pos_sub") or row.get("pos")
            hitters.append(row)

    for row in unknown_subs:
        row["slot"] = None
        row["indent"] = True
    hitters.extend(sorted(unknown_subs, key=lambda x: x["name"]))

    pitcher_order_index = {pid: idx for idx, pid in enumerate(pitchers_order)}
    pitchers.sort(key=lambda p: pitcher_order_index.get(p["id"], 9999))

    _save_player_cache(cache)
    return hitters, pitchers, {"orange_is_home": orange_is_home, "side_key": side_key}


def _prospect_info(player: Dict[str, Any], name_map: Dict[str, Dict[str, Any]], id_map: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pid = _safe_int(player.get("id"), -1)
    if pid in id_map:
        return id_map[pid]
    return name_map.get(_normalize_name(player.get("name", "")))


def _hitter_line(h: Dict[str, Any], with_slash: bool = True) -> str:
    base = f"{h.get('pos', '-') or '-'} {h['name']} {h['h']}-{h['ab']}"
    if with_slash and h.get("season_slash"):
        base += f" | {h['season_slash']}"
    return base


def _pitcher_line(p: Dict[str, Any], with_secondaries: bool = True) -> str:
    line = f"{p['hand']} {p['name']} {p['ip']} IP, {p['k']} K"
    if with_secondaries and (p.get("swstr") or p.get("gb")):
        parts = []
        if p.get("swstr"):
            parts.append(f"SwStr {p['swstr']}")
        if p.get("gb"):
            parts.append(f"GB {p['gb']}")
        line += f" ({', '.join(parts)})"
    return line


def compose_post_text(
    status: str,
    orange_runs: int,
    opp_name: str,
    opp_runs: int,
    hitters: List[Dict[str, Any]],
    pitchers: List[Dict[str, Any]],
) -> str:
    line1 = f"{status}: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}"
    name_map, id_map = _prospect_maps()

    candidates: List[Dict[str, Any]] = []
    for h in hitters:
        pinfo = _prospect_info(h, name_map, id_map)
        if pinfo:
            pr = _safe_int(pinfo.get("priority"), 9)
            candidates.append(
                {
                    "is_prospect": True,
                    "priority": pr,
                    "is_pitcher": False,
                    "perf": h["h"] * 3 + h["hr"] * 2 + h["bb"],
                    "full": _hitter_line(h, with_slash=True),
                    "no_slash": _hitter_line(h, with_slash=False),
                    "min": _hitter_line(h, with_slash=False),
                }
            )

    for p in pitchers:
        pinfo = _prospect_info(p, name_map, id_map)
        if pinfo:
            pr = _safe_int(pinfo.get("priority"), 9)
            candidates.append(
                {
                    "is_prospect": True,
                    "priority": pr,
                    "is_pitcher": True,
                    "perf": p["k"] * 2 + p["ip_outs"] / 3,
                    "full": _pitcher_line(p, with_secondaries=True),
                    "no_slash": _pitcher_line(p, with_secondaries=True),
                    "min": _pitcher_line(p, with_secondaries=False),
                }
            )

    chosen = sorted(candidates, key=lambda x: (x["priority"], -x["perf"], x["full"]))[:4]

    if any(_prospect_info(p, name_map, id_map) for p in pitchers) and not any(c["is_pitcher"] for c in chosen):
        prospect_pitchers = [p for p in pitchers if _prospect_info(p, name_map, id_map)]
        if prospect_pitchers:
            p = sorted(prospect_pitchers, key=lambda x: (-x["ip_outs"], -x["k"]))[0]
            chosen[-1:] = [
                {
                    "is_prospect": True,
                    "priority": 99,
                    "is_pitcher": True,
                    "perf": 0,
                    "full": _pitcher_line(p, with_secondaries=True),
                    "no_slash": _pitcher_line(p, with_secondaries=True),
                    "min": _pitcher_line(p, with_secondaries=False),
                }
            ]

    if pitchers and not any(c["is_pitcher"] for c in chosen):
        lead = sorted(pitchers, key=lambda p: (-p["ip_outs"], -p["k"]))[0]
        chosen.append(
            {
                "is_prospect": False,
                "priority": 99,
                "is_pitcher": True,
                "perf": 0,
                "full": _pitcher_line(lead, with_secondaries=True),
                "no_slash": _pitcher_line(lead, with_secondaries=True),
                "min": _pitcher_line(lead, with_secondaries=False),
            }
        )

    chosen = chosen[:4]

    for mode in ("full", "no_slash", "min"):
        lines = [line1] + [c[mode] for c in chosen]
        text = "\n".join(lines)
        if len(text) <= MAX_POST_CHARS:
            return text

    working = chosen[:]
    while working and len("\n".join([line1] + [c["min"] for c in working])) > MAX_POST_CHARS:
        nonpros = [i for i, c in enumerate(working) if not c["is_prospect"]]
        drop_idx = nonpros[-1] if nonpros else len(working) - 1
        working.pop(drop_idx)

    return "\n".join([line1] + [c["min"] for c in working])[:MAX_POST_CHARS]


def _load_tango_table() -> Dict[Tuple[int, str, int, int, int], float]:
    payload = _load_json(TANGO_WE_PATH, {"rows": []})
    table: Dict[Tuple[int, str, int, int, int], float] = {}
    for r in payload.get("rows", []):
        k = (_safe_int(r.get("inning"), 1), str(r.get("half") or "top").lower(), _safe_int(r.get("outs"), 0), _safe_int(r.get("base"), 0), _safe_int(r.get("scoreDiffHome"), 0))
        table[k] = _safe_float(r.get("weHome"), 0.5)
    return table


def _lookup_we(table: Dict[Tuple[int, str, int, int, int], float], inning: int, half: str, outs: int, base: int, score_diff_home: int) -> float:
    inning_c = min(max(inning, 1), 9)
    outs_c = min(max(outs, 0), 2)
    base_c = min(max(base, 0), 7)
    diff_c = min(max(score_diff_home, -10), 10)
    return table.get((inning_c, half.lower(), outs_c, base_c, diff_c), 0.5)


def _bases_from_runners(play: Dict[str, Any]) -> int:
    base = 0
    for r in play.get("runners") or []:
        movement = r.get("movement") or {}
        if movement.get("isOut"):
            continue
        end = str(movement.get("end") or "")
        if end == "1B":
            base |= 1
        elif end == "2B":
            base |= 2
        elif end == "3B":
            base |= 4
    return base


def _short_moment(play: Dict[str, Any], we_before: float, we_after: float, wpa: float) -> str:
    about = play.get("about") or {}
    result = play.get("result") or {}
    batter = ((play.get("matchup") or {}).get("batter") or {}).get("fullName") or "Player"
    surname = batter.split()[-1]
    event = (result.get("event") or "play").lower()
    runs = _safe_int(result.get("rbi"), 0)
    run_txt = f" {runs}-run" if runs > 1 else ""
    inning = _safe_int(about.get("inning"), 1)
    half = "T" if (about.get("halfInning") or "").lower() == "top" else "B"
    return f"{half}{inning}: {surname}{run_txt} {event} — WE {round(we_before*100):d}%→{round(we_after*100):d}% ({wpa*100:+.0f}%)"


def generate_key_moments_wpa(feed: Dict[str, Any], orange_is_home: bool) -> List[str]:
    table = _load_tango_table()
    all_plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []

    inning, half = 1, "top"
    outs_before = 0
    bases_before = 0
    home_score_before = 0
    away_score_before = 0
    debug = os.getenv("DEBUG_WPA", "0") == "1"
    debug_rows = []

    candidates: List[Tuple[float, str]] = []

    for play in all_plays:
        score_diff_before = home_score_before - away_score_before
        we_home_before = _lookup_we(table, inning, half, outs_before, bases_before, score_diff_before)
        we_orange_before = we_home_before if orange_is_home else (1 - we_home_before)

        about = play.get("about") or {}
        count = play.get("count") or {}
        result = play.get("result") or {}

        home_after = _safe_int(result.get("homeScore"), home_score_before)
        away_after = _safe_int(result.get("awayScore"), away_score_before)
        outs_after = _safe_int(count.get("outs"), outs_before)
        bases_after = _bases_from_runners(play)

        next_inning, next_half = inning, half
        lookup_outs_after, lookup_bases_after = outs_after, bases_after
        if outs_after >= 3:
            lookup_outs_after = 0
            lookup_bases_after = 0
            if half == "top":
                next_half = "bottom"
            else:
                next_half = "top"
                next_inning += 1

        score_diff_after = home_after - away_after
        we_home_after = _lookup_we(table, next_inning, next_half, lookup_outs_after, lookup_bases_after, score_diff_after)
        we_orange_after = we_home_after if orange_is_home else (1 - we_home_after)
        wpa = we_orange_after - we_orange_before

        if abs(wpa) > 0.001:
            candidates.append((abs(wpa), _short_moment(play, we_orange_before, we_orange_after, wpa)))

        if debug and len(debug_rows) < 8:
            debug_rows.append(
                f"{half[0].upper()}{inning} outs:{outs_before}->{outs_after} base:{bases_before}->{bases_after} diff:{score_diff_before}->{score_diff_after} we:{we_orange_before:.3f}->{we_orange_after:.3f}"
            )

        inning, half = next_inning, next_half
        outs_before = lookup_outs_after
        bases_before = lookup_bases_after
        home_score_before = home_after
        away_score_before = away_after

    if debug:
        print("DEBUG_WPA samples:")
        for row in debug_rows:
            print(f"  {row}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    moments = [c[1] for c in candidates[:3]]
    while len(moments) < 3:
        moments.append("No high-leverage plate appearance found")
    return moments


def _prospect_name_ids(hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> set:
    name_map, id_map = _prospect_maps()
    ids = set()
    for h in hitters + pitchers:
        if _prospect_info(h, name_map, id_map):
            ids.add(h.get("id"))
    return ids


def _to_html_table_rows_hitters(hitters: List[Dict[str, Any]], prospect_ids: set) -> str:
    rows = []
    for h in hitters:
        classes = "sub" if h.get("indent") else ""
        name_cls = "hl" if h.get("id") in prospect_ids else ""
        display_name = f"↳ {h['name']}" if h.get("indent") else h["name"]
        rows.append(
            f"<tr class='{classes}'><td class='name {name_cls}'>{html.escape(display_name)}</td>"
            f"<td>{html.escape(h.get('pos','-'))}</td><td>{h['pa']}</td><td>{h['ab']}</td><td>{h['r']}</td><td>{h['h']}</td>"
            f"<td>{h['bb']}</td><td>{h['k']}</td><td>{h['2b']}</td><td>{h['3b']}</td><td>{h['hr']}</td><td>{h['sb']}</td><td>{h['tb']}</td>"
            f"<td>{html.escape(h.get('season_slash') or '')}</td></tr>"
        )
    return "\n".join(rows)


def _to_html_table_rows_pitchers(pitchers: List[Dict[str, Any]], prospect_ids: set) -> str:
    rows = []
    for p in pitchers:
        name_cls = "hl" if p.get("id") in prospect_ids else ""
        rows.append(
            f"<tr><td class='name {name_cls}'>{html.escape(p['name'])}</td><td>{p['hand']}</td><td>{p['ip']}</td><td>{p['h']}</td><td>{p['r']}</td>"
            f"<td>{p['er']}</td><td>{p['bb']}</td><td>{p['k']}</td><td>{p['bf']}</td><td>{p['whip']}</td><td>{p.get('swstr','')}</td><td>{p.get('gb','')}</td></tr>"
        )
    return "\n".join(rows)


def _render_html_card(
    matchup: str,
    game_date: str,
    status: str,
    score_line: str,
    linescore: str,
    hitters: List[Dict[str, Any]],
    pitchers: List[Dict[str, Any]],
    key_moments: List[str],
) -> str:
    prospect_ids = _prospect_name_ids(hitters, pitchers)
    hitter_rows = _to_html_table_rows_hitters(hitters, prospect_ids)
    pitcher_rows = _to_html_table_rows_pitchers(pitchers, prospect_ids)
    moment_items = "".join(f"<li>{html.escape(m)}</li>" for m in key_moments[:3])

    return f"""
<!doctype html><html><head><meta charset='utf-8'/>
<style>
body {{ margin:0; padding:0; background:white; }}
#card {{ width: 980px; padding: 20px 24px 18px 24px; font-family: Georgia, 'Times New Roman', serif; color:#141414; }}
.hdr1 {{ font-size: 30px; font-weight: 700; line-height: 1.05; margin: 0 0 4px 0; }}
.hdr2 {{ font-size: 15px; letter-spacing: .03em; color:#333; margin-bottom: 4px; }}
.hdr3 {{ font-size: 20px; font-weight: 700; margin-bottom: 6px; }}
.rule {{ border-top: 1px solid #1d1d1d; margin: 6px 0 10px 0; }}
.section {{ margin-top: 10px; }}
.section h3 {{ font-variant: small-caps; letter-spacing: .07em; font-size: 15px; margin: 0 0 4px 0; border-bottom:1px solid #ccc; padding-bottom:2px; }}
table {{ width:100%; border-collapse: collapse; font-size: 13px; line-height:1.15; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 2px 4px; text-align: right; border-bottom: 1px solid #efefef; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
tr:nth-child(even) td {{ background: #fcfcfc; }}
.name.subtext {{ color:#555; }}
tr.sub td {{ color:#666; font-size:12px; }}
tr.sub td.name {{ padding-left: 14px; }}
.hl {{ position:relative; z-index:0; display:inline-block; }}
.hl::before {{ content:''; position:absolute; left:-2px; right:-2px; top:55%; height:0.95em; background:rgba(255,235,59,0.55); transform:rotate(-1.2deg); border-radius:2px; z-index:-1; }}
.moments ol {{ margin: 4px 0 0 18px; padding:0; }}
.moments li {{ margin: 2px 0; font-size: 13px; line-height:1.2; }}
</style></head><body>
<div id='card'>
  <div class='hdr1'>{html.escape(matchup)}</div>
  <div class='hdr2'>{html.escape(game_date)} · {html.escape(status)}</div>
  <div class='hdr3'>{html.escape(score_line)}</div>
  <div class='hdr2'>{html.escape(linescore)}</div>
  <div class='rule'></div>

  <div class='section'>
    <h3>Hitters (all appearances)</h3>
    <table>
      <thead><tr><th>Player</th><th>Pos</th><th>PA</th><th>AB</th><th>R</th><th>H</th><th>BB</th><th>K</th><th>2B</th><th>3B</th><th>HR</th><th>SB</th><th>TB</th><th>Season</th></tr></thead>
      <tbody>{hitter_rows}</tbody>
    </table>
  </div>

  <div class='section'>
    <h3>Pitchers (all appearances)</h3>
    <table>
      <thead><tr><th>Pitcher</th><th>Hand</th><th>IP</th><th>H</th><th>R</th><th>ER</th><th>BB</th><th>K</th><th>BF</th><th>WHIP</th><th>SwStr</th><th>GB</th></tr></thead>
      <tbody>{pitcher_rows}</tbody>
    </table>
  </div>

  <div class='section moments'>
    <h3>Key moments (WPA via Tango WE table)</h3>
    <ol>{moment_items}</ol>
  </div>
</div>
</body></html>
"""


def render_boxscore_card_image(
    output_path: str,
    matchup: str,
    game_date: str,
    status: str,
    score_line: str,
    linescore: str,
    hitters: List[Dict[str, Any]],
    pitchers: List[Dict[str, Any]],
    key_moments: List[str],
) -> str:
    html_doc = _render_html_card(matchup, game_date, status, score_line, linescore, hitters, pitchers, key_moments)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1200, "height": 4200})
            page.set_content(html_doc, wait_until="load")
            card = page.locator("#card")
            box = card.bounding_box()
            clip = {
                "x": max(0, box["x"] - 2),
                "y": max(0, box["y"] - 2),
                "width": box["width"] + 4,
                "height": box["height"] + 4,
            }
            quality = 88
            page.screenshot(path=output_path, type="jpeg", quality=quality, clip=clip)
            while os.path.getsize(output_path) > 1_000_000 and quality > 45:
                quality -= 7
                page.screenshot(path=output_path, type="jpeg", quality=quality, clip=clip)
            browser.close()
            return output_path
    except Exception as exc:
        print(f"WARNING: Playwright render failed ({exc}); using Pillow fallback")

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1080, 1600), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default()
    y = 20
    for line in [matchup, f"{game_date} {status}", score_line, linescore, "", "Render fallback active"]:
        d.text((20, y), line, fill=(0, 0, 0), font=f)
        y += 22
    img.save(output_path, format="JPEG", quality=80, optimize=True)
    return output_path


def post_to_bluesky_with_image(client: Client, text: str, image_path: str, alt_text: str) -> None:
    with open(image_path, "rb") as f:
        blob_resp = client.upload_blob(f.read())
    image_blob = blob_resp.get("blob") if isinstance(blob_resp, dict) else getattr(blob_resp, "blob", blob_resp)
    embed = models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(alt=alt_text, image=image_blob)])
    client.send_post(text=text, embed=embed)
    time.sleep(SLEEP_BETWEEN_POSTS_SEC)


def _most_recent_gamepk_from_window(session: requests.Session, lookback_days: int = 14) -> Optional[int]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    games = fetch_schedule_games(session, start_date=start.isoformat(), end_date=end.isoformat())
    dated = sorted([(g.get("officialDate") or "", _safe_int(g.get("gamePk"))) for g in games if g.get("gamePk")], reverse=True)
    return dated[0][1] if dated else None


def _get_game_from_overrides(session: requests.Session) -> List[int]:
    override_gamepk = (os.getenv("OVERRIDE_GAMEPK") or "").strip()
    override_date = (os.getenv("OVERRIDE_DATE") or "").strip()
    if override_gamepk:
        print(f"Using OVERRIDE_GAMEPK={override_gamepk}")
        return [_safe_int(override_gamepk)]
    if override_date:
        print(f"Using OVERRIDE_DATE={override_date}")
        return [_safe_int(g.get("gamePk")) for g in fetch_schedule_games(session, target_date=override_date) if g.get("gamePk")]
    if (os.getenv("GITHUB_EVENT_NAME") or "") == "workflow_dispatch":
        recent = _most_recent_gamepk_from_window(session)
        if recent:
            print(f"No override provided on workflow_dispatch; using most recent gamePk={recent}")
            return [recent]
    return []


def _orange_score_tuple(feed: Dict[str, Any], orange_is_home: bool) -> Tuple[int, int]:
    teams = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
    home = _safe_int(((teams.get("home") or {}).get("runs") or 0))
    away = _safe_int(((teams.get("away") or {}).get("runs") or 0))
    return (home, away) if orange_is_home else (away, home)


def _should_post_game(entry: Dict[str, Any], status: str, force_repost: bool = False) -> bool:
    if force_repost:
        return True
    if status == "Suspended" and not entry.get("posted_suspended", False):
        return True
    if status == "Final" and not entry.get("posted_final", False):
        return True
    return False


def update_state(state: Dict[str, Any], game_pk: int, status: str, posted: bool) -> None:
    key = str(game_pk)
    entry = state.setdefault("posted_games", {}).setdefault(key, {"posted_suspended": False, "posted_final": False, "last_status": None, "last_seen_iso": None})
    if posted and status == "Suspended":
        entry["posted_suspended"] = True
    if posted and status == "Final":
        entry["posted_final"] = True
    entry["last_status"] = status
    entry["last_seen_iso"] = datetime.now(timezone.utc).isoformat()


def run_finals_mode() -> None:
    state = load_state()
    session = make_session()
    maybe_warn_stale_prospects()

    game_pks = _get_game_from_overrides(session)
    if not game_pks:
        game_pks = [_safe_int(g.get("gamePk")) for g in fetch_schedule_games(session) if g.get("gamePk")]
    if not game_pks:
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print("No DSL Orange games in schedule window.")
        return

    dry_run = os.getenv("DRY_RUN", "0") == "1"
    force_repost = os.getenv("FORCE_REPOST", "0") == "1"
    client: Optional[Client] = None
    posted_count = 0

    for game_pk in game_pks:
        feed = fetch_game_feed(session, game_pk)
        if not feed:
            continue
        status = classify_terminal_status(feed)
        if status not in {"Final", "Suspended"}:
            continue

        game_data = feed.get("gameData") or {}
        teams = game_data.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        orange_is_home = home.get("id") == DSL_ORANGE_TEAM_ID
        opp_name = away.get("name", "Opponent") if orange_is_home else home.get("name", "Opponent")

        entry = state.setdefault("posted_games", {}).setdefault(str(game_pk), {"posted_suspended": False, "posted_final": False, "last_status": None, "last_seen_iso": None})
        if not _should_post_game(entry, status, force_repost=force_repost):
            update_state(state, game_pk, status, posted=False)
            continue

        hitters, pitchers, _ = extract_player_lines(feed, session)
        key_moments = generate_key_moments_wpa(feed, orange_is_home=orange_is_home)

        orange_runs, opp_runs = _orange_score_tuple(feed, orange_is_home)
        linescore_obj = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
        home_line, away_line = linescore_obj.get("home") or {}, linescore_obj.get("away") or {}
        linescore = (
            f"R/H/E — {away.get('abbreviation', 'AWY')} {away_line.get('runs', '-')}/{away_line.get('hits', '-')}/{away_line.get('errors', '-')}"
            f" | {home.get('abbreviation', 'HME')} {home_line.get('runs', '-')}/{home_line.get('hits', '-')}/{home_line.get('errors', '-')}"
        )

        image_path = f"dsl_orange_{game_pk}_{status.lower()}.jpg"
        render_boxscore_card_image(
            output_path=image_path,
            matchup=f"{away.get('name', 'Away')} at {home.get('name', 'Home')}",
            game_date=((game_data.get("datetime") or {}).get("officialDate") or datetime.now(timezone.utc).date().isoformat()),
            status=status,
            score_line=f"DSL Giants Orange {orange_runs} - {opp_runs} {opp_name}",
            linescore=linescore,
            hitters=hitters,
            pitchers=pitchers,
            key_moments=key_moments,
        )

        text = compose_post_text(status, orange_runs, opp_name, opp_runs, hitters, pitchers)
        alt_text = f"{status} box score: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}."

        if dry_run:
            print(f"DRY_RUN: would post {status} for game {game_pk}:\n{text}")
            posted_count += 1
            update_state(state, game_pk, status, posted=True)
            continue

        if client is None:
            client = bsky_login()
        post_to_bluesky_with_image(client, text, image_path, alt_text)
        posted_count += 1
        update_state(state, game_pk, status, posted=True)

    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"Processed {len(game_pks)} games, posted {posted_count} updates.")


def run_daily_recap_mode() -> None:
    state = load_state()
    session = make_session()
    target_day = os.getenv("RECAP_DATE") or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    if state.setdefault("recaps", {}).get(target_day):
        print(f"Recap already posted for {target_day}.")
        return

    games = fetch_schedule_games(session, target_date=target_day)
    if not games:
        print(f"No games on {target_day}.")
        state["recaps"][target_day] = {"posted": False, "note": "no_games"}
        save_state(state)
        return

    finals: List[str] = []
    wins = losses = 0
    for g in games:
        feed = fetch_game_feed(session, _safe_int(g.get("gamePk")))
        if not feed or classify_terminal_status(feed) != "Final":
            continue
        gd = feed.get("gameData") or {}
        teams = gd.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        orange_is_home = home.get("id") == DSL_ORANGE_TEAM_ID
        opp_name = away.get("name", "Opponent") if orange_is_home else home.get("name", "Opponent")
        orange_runs, opp_runs = _orange_score_tuple(feed, orange_is_home)
        wins += 1 if orange_runs > opp_runs else 0
        losses += 1 if orange_runs < opp_runs else 0
        finals.append(f"Final: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}")

    if not finals:
        print(f"No final games to recap on {target_day}.")
        state["recaps"][target_day] = {"posted": False, "note": "no_finals"}
        save_state(state)
        return

    text = (f"Yesterday: DSL Orange went {wins}-{losses}. " + " | ".join(finals[:2]))[:MAX_POST_CHARS]
    client = bsky_login()
    client.send_post(text=text)
    time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    state["recaps"][target_day] = {"posted": True, "at": datetime.now(timezone.utc).isoformat()}
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSL Giants Orange gameday poster")
    parser.add_argument("--recap", action="store_true", help="Run daily recap mode")
    args = parser.parse_args()
    run_daily_recap_mode() if args.recap else run_finals_mode()


if __name__ == "__main__":
    main()
