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
#   DRY_RUN=1 OVERRIDE_GAMEPK=811804 FORCE_REPOST=1 DEBUG_WPA=1 python gameday_dsl_orange.py

API_BASE = "https://statsapi.mlb.com"
DSL_ORANGE_TEAM_ID = 615
SPORT_ID = 16
MAX_POST_CHARS = 300
PROSPECTS_PATH = Path("prospects.json")
PROSPECTS_STALE_DAYS = 45
PLAYER_CACHE_PATH = Path("player_cache.json")
TANGO_WE_PATH = Path("data/tango_we.json")
CARD_TEMPLATE_PATH = Path("templates/boxscore_card.html")


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


def _load_prospects() -> Dict[str, Any]:
    return _load_json(PROSPECTS_PATH, {"updated": "", "prospects": []})


def _prospect_maps() -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    name_map: Dict[str, Dict[str, Any]] = {}
    id_map: Dict[int, Dict[str, Any]] = {}
    for p in _load_prospects().get("prospects", []):
        nm = _normalize_name(p.get("name", ""))
        if nm:
            name_map[nm] = p
        if p.get("personId") is not None:
            id_map[_safe_int(p.get("personId"))] = p
    return name_map, id_map


def _load_player_cache() -> Dict[str, Any]:
    return _load_json(PLAYER_CACHE_PATH, {"players": {}})


def _save_player_cache(cache: Dict[str, Any]) -> None:
    _save_json(PLAYER_CACHE_PATH, cache)


def _extract_season_era_from_people_payload(payload: Dict[str, Any]) -> str:
    people = payload.get("people") or []
    if not people:
        return ""
    stats = people[0].get("stats") or []
    for grp in stats:
        splits = grp.get("splits") or []
        if not splits:
            continue
        stat = splits[0].get("stat") or {}
        era = stat.get("era")
        if era:
            return str(era)
    return ""


def _get_person_details(session: requests.Session, person_id: int, cache: Dict[str, Any]) -> Dict[str, Any]:
    key = str(person_id)
    players = cache.setdefault("players", {})
    if key in players:
        return players[key]

    detail = {"pitchHand": "", "seasonEra": ""}
    try:
        res = session.get(
            f"{API_BASE}/api/v1/people/{person_id}",
            params={"hydrate": "stats(group=[pitching],type=[season],sportId=16)"},
            timeout=30,
        )
        res.raise_for_status()
        payload = res.json() or {}
        person = (payload.get("people") or [{}])[0]
        detail["pitchHand"] = ((person.get("pitchHand") or {}).get("code") or "").upper()
        detail["seasonEra"] = _extract_season_era_from_people_payload(payload)
    except requests.RequestException:
        pass

    players[key] = detail
    return detail


def maybe_warn_stale_prospects() -> None:
    if not PROSPECTS_PATH.exists():
        print("WARNING: prospects.json not found")
        return
    mtime = datetime.fromtimestamp(PROSPECTS_PATH.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    if age_days > PROSPECTS_STALE_DAYS:
        print(f"WARNING: prospects.json is {age_days} days old; consider refreshing rankings.")


def _player_season_slash(player: Dict[str, Any]) -> Optional[str]:
    season_bat = ((player.get("seasonStats") or {}).get("batting") or {})
    avg, obp, slg = season_bat.get("avg"), season_bat.get("obp"), season_bat.get("slg")
    if avg and obp and slg:
        return f"{avg}/{obp}/{slg}"
    return None


def _hitter_season_compact(player: Dict[str, Any]) -> str:
    season_bat = ((player.get("seasonStats") or {}).get("batting") or {})
    tokens = []
    pa = season_bat.get("plateAppearances")
    ops = season_bat.get("ops")
    hr = season_bat.get("homeRuns")
    sb = season_bat.get("stolenBases")
    if pa not in (None, ""):
        tokens.append(f"PA {pa}")
    if ops not in (None, ""):
        tokens.append(f"OPS {ops}")
    if hr not in (None, ""):
        tokens.append(f"HR {hr}")
    if sb not in (None, ""):
        tokens.append(f"SB {sb}")
    return " ".join(tokens)


def _pitcher_season_metrics(pdata: Dict[str, Any], people_era: str) -> Dict[str, str]:
    season_p = ((pdata.get("seasonStats") or {}).get("pitching") or {})
    era = str(season_p.get("era") or people_era or "")

    k = _safe_float(season_p.get("strikeOuts"), 0.0)
    bb = _safe_float(season_p.get("baseOnBalls"), 0.0)
    bf = _safe_float(season_p.get("battersFaced"), 0.0)
    ip = str(season_p.get("inningsPitched") or "0.0")
    outs = _ip_to_outs(ip)
    ip_val = outs / 3.0 if outs > 0 else 0.0

    k_pct = f"{(k / bf) * 100:.1f}" if bf > 0 else ""
    bb_pct = f"{(bb / bf) * 100:.1f}" if bf > 0 else ""
    k9 = f"{(k / ip_val) * 9:.1f}" if ip_val > 0 else ""
    bb9 = f"{(bb / ip_val) * 9:.1f}" if ip_val > 0 else ""

    return {"era": era, "k_pct": k_pct, "bb_pct": bb_pct, "k9": k9, "bb9": bb9}

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


def _is_pitcher_only(all_positions: List[Dict[str, Any]]) -> bool:
    if not all_positions:
        return False
    vals = [p.get("abbreviation") for p in all_positions if p.get("abbreviation")]
    return bool(vals) and all(v == "P" for v in vals)


def _primary_non_pitcher_pos(all_positions: List[Dict[str, Any]]) -> str:
    for p in all_positions:
        ab = p.get("abbreviation") or ""
        if ab and ab != "P":
            return ab
    return "-"


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
        game_status = pdata.get("gameStatus") or {}
        batting = ((pdata.get("stats") or {}).get("batting") or {})
        pitching = ((pdata.get("stats") or {}).get("pitching") or {})
        bo_raw = str(pdata.get("battingOrder") or "").strip()

        pa = _safe_int(batting.get("plateAppearances"), _safe_int(batting.get("atBats")) + _safe_int(batting.get("baseOnBalls")))
        appeared_hitter = bool(bo_raw) or bool(all_positions) or (game_status.get("isOnBench") is False) or pid in batters_set

        if appeared_hitter and _is_pitcher_only(all_positions) and pa <= 0:
            appeared_hitter = False

        if appeared_hitter:
            doubles = _safe_int(batting.get("doubles"))
            triples = _safe_int(batting.get("triples"))
            hr = _safe_int(batting.get("homeRuns"))
            hits = _safe_int(batting.get("hits"))
            tb = _safe_int(batting.get("totalBases"), hits + doubles + 2 * triples + 3 * hr)
            row = {
                "id": pid,
                "name": name,
                "pos": _primary_non_pitcher_pos(all_positions),
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
                "season_slash": _player_season_slash(pdata) or "",
                "season_compact": _hitter_season_compact(pdata),
                "indent": False,
                "slot": None,
                "seq": 999,
            }
            if bo_raw.isdigit():
                bo = int(bo_raw)
                slot, seq = bo // 100, bo % 100
                row["slot"] = slot
                row["seq"] = seq
                if 1 <= slot <= 9:
                    slot_groups[slot].append(row)
                else:
                    unknown_subs.append(row)
            else:
                unknown_subs.append(row)

        appeared_pitcher = pid in set(pitchers_order) or bool(pitching) or any(k in pitching for k in ("inningsPitched", "numberOfPitches", "strikeOuts"))
        if appeared_pitcher:
            meta = _get_person_details(session, pid, cache)
            hand = "RHP" if meta.get("pitchHand") == "R" else "LHP" if meta.get("pitchHand") == "L" else "P"
            ip = str(pitching.get("inningsPitched") or "0.0")
            pm = pitch_metrics.get(pid, {})
            season = _pitcher_season_metrics(pdata, meta.get("seasonEra") or "")
            pitchers.append(
                {
                    "id": pid,
                    "name": name,
                    "hand": hand,
                    "ip": ip,
                    "ip_outs": _ip_to_outs(ip),
                    "h": _safe_int(pitching.get("hits")),
                    "r": _safe_int(pitching.get("runs")),
                    "er": _safe_int(pitching.get("earnedRuns")),
                    "bb": _safe_int(pitching.get("baseOnBalls")),
                    "k": _safe_int(pitching.get("strikeOuts")),
                    "bf": _safe_int(pitching.get("battersFaced")),
                    "season_era": season["era"],
                    "season_k_pct": season["k_pct"],
                    "season_bb_pct": season["bb_pct"],
                    "season_k9": season["k9"],
                    "season_bb9": season["bb9"],
                    "swstr": pm.get("swstr") or "",
                    "gb": pm.get("gb") or "",
                }
            )

    hitters: List[Dict[str, Any]] = []
    for slot in range(1, 10):
        rows = sorted(slot_groups[slot], key=lambda r: (r["seq"], r["name"]))
        for idx, row in enumerate(rows):
            row["indent"] = idx > 0
            hitters.append(row)
    for row in sorted(unknown_subs, key=lambda r: r["name"]):
        row["indent"] = True
        hitters.append(row)

    order_map = {pid: idx for idx, pid in enumerate(pitchers_order)}
    pitchers.sort(key=lambda p: order_map.get(p["id"], 9999))
    _save_player_cache(cache)

    return hitters, pitchers, {"orange_is_home": orange_is_home, "side_key": side_key}

def _prospect_info(player: Dict[str, Any], name_map: Dict[str, Dict[str, Any]], id_map: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pid = _safe_int(player.get("id"), -1)
    if pid in id_map:
        return id_map[pid]
    return name_map.get(_normalize_name(player.get("name", "")))


def _hitter_text_line(h: Dict[str, Any], include_slash: bool = True) -> str:
    line = f"{h.get('pos','-')} {h['name']} {h['h']}-{h['ab']}"
    if include_slash and h.get("season_slash"):
        line += f" | {h['season_slash']}"
    return line


def _pitcher_text_line(p: Dict[str, Any], include_secondaries: bool = True) -> str:
    line = f"{p['hand']} {p['name']} {p['ip']} IP, {p['k']} K"
    if include_secondaries and (p.get("swstr") or p.get("gb")):
        parts = []
        if p.get("swstr"):
            parts.append(f"SwStr {p['swstr']}")
        if p.get("gb"):
            parts.append(f"GB {p['gb']}")
        line += f" ({', '.join(parts)})"
    return line


def compose_post_text(status: str, orange_runs: int, opp_name: str, opp_runs: int, hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> str:
    first = f"{status}: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}"
    name_map, id_map = _prospect_maps()

    lines: List[Dict[str, Any]] = []
    for h in hitters:
        p = _prospect_info(h, name_map, id_map)
        if p:
            lines.append(
                {
                    "priority": _safe_int(p.get("priority"), 9),
                    "pitcher": False,
                    "perf": h["h"] * 3 + h["hr"] * 2 + h["bb"],
                    "full": _hitter_text_line(h, include_slash=True),
                    "compact": _hitter_text_line(h, include_slash=False),
                }
            )
    for p in pitchers:
        pp = _prospect_info(p, name_map, id_map)
        if pp:
            lines.append(
                {
                    "priority": _safe_int(pp.get("priority"), 9),
                    "pitcher": True,
                    "perf": p["k"] * 2 + p["ip_outs"] / 3,
                    "full": _pitcher_text_line(p, include_secondaries=True),
                    "compact": _pitcher_text_line(p, include_secondaries=False),
                }
            )

    chosen = sorted(lines, key=lambda x: (x["priority"], -x["perf"], x["full"]))[:4]

    if pitchers and not any(c["pitcher"] for c in chosen):
        lead = sorted(pitchers, key=lambda p: (-p["ip_outs"], -p["k"]))[0]
        chosen.append(
            {
                "priority": 99,
                "pitcher": True,
                "perf": 0,
                "full": _pitcher_text_line(lead, include_secondaries=True),
                "compact": _pitcher_text_line(lead, include_secondaries=False),
                "is_non_prospect": True,
            }
        )

    chosen = chosen[:4]

    # First pass: rich format.
    text = "\n".join([first] + [c["full"] for c in chosen])
    if len(text) <= MAX_POST_CHARS:
        return text

    # Remove non-prospect fallback first.
    chosen = [c for c in chosen if not c.get("is_non_prospect")]
    text = "\n".join([first] + [c["full"] for c in chosen])
    if len(text) <= MAX_POST_CHARS:
        return text

    # Remove slash/secondary metrics.
    text = "\n".join([first] + [c["compact"] for c in chosen])
    if len(text) <= MAX_POST_CHARS:
        return text

    # Drop from bottom until fit.
    working = chosen[:]
    while working and len("\n".join([first] + [c["compact"] for c in working])) > MAX_POST_CHARS:
        working.pop()
    return "\n".join([first] + [c["compact"] for c in working])[:MAX_POST_CHARS]


def _load_tango_table() -> Dict[Tuple[int, str, int, int, int], float]:
    payload = _load_json(TANGO_WE_PATH, {"rows": []})
    table: Dict[Tuple[int, str, int, int, int], float] = {}
    for row in payload.get("rows", []):
        key = (
            _safe_int(row.get("inning"), 1),
            str(row.get("half") or "top").lower(),
            _safe_int(row.get("outs"), 0),
            _safe_int(row.get("base"), 0),
            _safe_int(row.get("scoreDiffHome"), 0),
        )
        table[key] = _safe_float(row.get("weHome"), 0.5)
    return table


def _lookup_we(table: Dict[Tuple[int, str, int, int, int], float], inning: int, half: str, outs: int, base: int, score_diff_home: int) -> float:
    inning = min(max(inning, 1), 9)
    outs = min(max(outs, 0), 2)
    base = min(max(base, 0), 7)
    score_diff_home = min(max(score_diff_home, -10), 10)
    return table.get((inning, half, outs, base, score_diff_home), 0.5)


def _bases_from_runners(play: Dict[str, Any]) -> int:
    mask = 0
    for r in play.get("runners") or []:
        move = r.get("movement") or {}
        if move.get("isOut"):
            continue
        end = str(move.get("end") or "")
        if end == "1B":
            mask |= 1
        elif end == "2B":
            mask |= 2
        elif end == "3B":
            mask |= 4
    return mask


def _moment_text(play: Dict[str, Any], we_before: float, we_after: float, wpa: float) -> str:
    about = play.get("about") or {}
    result = play.get("result") or {}
    batter = ((play.get("matchup") or {}).get("batter") or {}).get("fullName") or "Player"
    surname = batter.split()[-1]
    event = (result.get("event") or "play").lower()
    runs = _safe_int(result.get("rbi"), 0)
    run_prefix = f" {runs}-run" if runs > 1 else ""
    inning = _safe_int(about.get("inning"), 1)
    half = "T" if (about.get("halfInning") or "").lower() == "top" else "B"
    return f"{half}{inning}: {surname}{run_prefix} {event} — Orange WE {round(we_before*100):d}%→{round(we_after*100):d}% ({wpa*100:+.0f}%)"


def generate_key_moments_wpa(feed: Dict[str, Any], orange_is_home: bool) -> List[str]:
    table = _load_tango_table()
    plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []
    debug = os.getenv("DEBUG_WPA", "0") == "1"

    inning, half = 1, "top"
    outs_before = 0
    base_before = 0
    home_score = 0
    away_score = 0

    candidates: List[Tuple[float, str, Dict[str, Any]]] = []

    for play in plays:
        diff_before = home_score - away_score
        we_home_before = _lookup_we(table, inning, half, outs_before, base_before, diff_before)
        we_orange_before = we_home_before if orange_is_home else (1.0 - we_home_before)

        result = play.get("result") or {}
        count = play.get("count") or {}

        home_after = _safe_int(result.get("homeScore"), home_score)
        away_after = _safe_int(result.get("awayScore"), away_score)
        outs_after = _safe_int(count.get("outs"), outs_before)
        base_after = _bases_from_runners(play)

        next_inning, next_half = inning, half
        look_outs, look_base = outs_after, base_after
        if outs_after >= 3:
            look_outs, look_base = 0, 0
            if half == "top":
                next_half = "bottom"
            else:
                next_half = "top"
                next_inning += 1

        diff_after = home_after - away_after
        we_home_after = _lookup_we(table, next_inning, next_half, look_outs, look_base, diff_after)
        we_orange_after = we_home_after if orange_is_home else (1.0 - we_home_after)
        wpa = we_orange_after - we_orange_before

        if abs(wpa) > 0.001:
            detail = {
                "inning": inning,
                "half": half,
                "outs_before": outs_before,
                "outs_after": outs_after,
                "base_before": base_before,
                "base_after": base_after,
                "diff_before": diff_before,
                "diff_after": diff_after,
                "we_home_before": we_home_before,
                "we_home_after": we_home_after,
                "we_orange_before": we_orange_before,
                "we_orange_after": we_orange_after,
                "wpa": wpa,
                "text": _moment_text(play, we_orange_before, we_orange_after, wpa),
            }
            candidates.append((abs(wpa), detail["text"], detail))

        inning, half = next_inning, next_half
        outs_before, base_before = look_outs, look_base
        home_score, away_score = home_after, away_after

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[:3]

    if debug:
        print("DEBUG_WPA selected moments:")
        for _, _, d in picked:
            print(
                "  "
                f"{d['half'][0].upper()}{d['inning']} outs:{d['outs_before']}->{d['outs_after']} "
                f"base:{d['base_before']}->{d['base_after']} scoreDiffHome:{d['diff_before']}->{d['diff_after']} "
                f"WE_home:{d['we_home_before']:.3f}->{d['we_home_after']:.3f} "
                f"WE_orange:{d['we_orange_before']:.3f}->{d['we_orange_after']:.3f} WPA:{d['wpa']:+.3f}"
            )

    moments = [p[1] for p in picked]
    while len(moments) < 3:
        moments.append("No high-leverage plate appearance found")
    return moments

def _prospect_ids(hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> set:
    name_map, id_map = _prospect_maps()
    ids = set()
    for pl in hitters + pitchers:
        if _prospect_info(pl, name_map, id_map):
            ids.add(pl.get("id"))
    return ids


def _build_hitter_rows(hitters: List[Dict[str, Any]], prospect_ids: set) -> str:
    rows = []
    for h in hitters:
        row_cls = "sub" if h.get("indent") else ""
        prefix = html.escape(h.get("pos", "-"))
        name = html.escape(h["name"])
        hl_cls = "hl" if h.get("id") in prospect_ids else ""
        indent_cls = "indent" if h.get("indent") else ""

        xbh_bits = []
        if h.get("2b", 0) > 0:
            xbh_bits.append(f"{h['2b']} 2B")
        if h.get("3b", 0) > 0:
            xbh_bits.append(f"{h['3b']} 3B")
        if h.get("hr", 0) > 0:
            xbh_bits.append(f"{h['hr']} HR")
        xbh = f"<span class='xbh'> ({html.escape(', '.join(xbh_bits))})</span>" if xbh_bits else ""

        player_cell = f"<span class='name-wrap {indent_cls}'>{prefix} <span class='{hl_cls}'>{name}</span></span>"
        h_cell = f"<span class='h'>{h['h']}</span>{xbh}"

        rows.append(
            f"<tr class='{row_cls}'><td class='player'>{player_cell}</td>"
            f"<td class='num'>{h['pa']}</td><td class='num'>{h['ab']}</td><td class='num'>{h['r']}</td><td class='num hcell'>{h_cell}</td>"
            f"<td class='num'>{h['bb']}</td><td class='num'>{h['k']}</td><td class='num'>{h['sb']}</td><td class='season'>{html.escape(h.get('season_compact',''))}</td></tr>"
        )
    return "\n".join(rows)

def _build_pitcher_rows(pitchers: List[Dict[str, Any]], prospect_ids: set) -> str:
    rows = []
    for p in pitchers:
        name = html.escape(p["name"])
        hl_cls = "hl" if p.get("id") in prospect_ids else ""
        first_cell = f"{p['hand']} <span class='{hl_cls}'>{name}</span>"

        season_tokens = []
        if p.get("season_era"):
            season_tokens.append(f"ERA {p['season_era']}")
        if p.get("season_k_pct"):
            season_tokens.append(f"K% {p['season_k_pct']}")
        if p.get("season_bb_pct"):
            season_tokens.append(f"BB% {p['season_bb_pct']}")

        lite_tokens = []
        if p.get("season_k9"):
            lite_tokens.append(f"k9 {p['season_k9']}")
        if p.get("season_bb9"):
            lite_tokens.append(f"bb9 {p['season_bb9']}")

        season_html = html.escape(" ".join(season_tokens))
        if lite_tokens:
            season_html += f" <span class='lite'>{html.escape(' '.join(lite_tokens))}</span>"

        rows.append(
            f"<tr><td class='player'>{first_cell}</td><td class='num'>{p['ip']}</td><td class='num'>{p['h']}</td><td class='num'>{p['r']}</td><td class='num'>{p['er']}</td><td class='num'>{p['bb']}</td><td class='num'>{p['k']}</td>"
            f"<td class='num'>{p['bf']}</td><td class='num'>{p.get('swstr','')}</td><td class='num'>{p.get('gb','')}</td><td class='season'>{season_html}</td></tr>"
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
    tpl = CARD_TEMPLATE_PATH.read_text(encoding="utf-8")
    pids = _prospect_ids(hitters, pitchers)
    payload = {
        "__MATCHUP__": html.escape(matchup),
        "__DATE_STATUS__": html.escape(f"{game_date} · {status}"),
        "__SCORE_LINE__": html.escape(score_line),
        "__RH_E__": html.escape(linescore),
        "__HITTER_ROWS__": _build_hitter_rows(hitters, pids),
        "__PITCHER_ROWS__": _build_pitcher_rows(pitchers, pids),
        "__MOMENT_ROWS__": "".join(f"<li>{html.escape(m)}</li>" for m in key_moments[:3]),
    }
    for k, v in payload.items():
        tpl = tpl.replace(k, v)
    return tpl


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
            page = browser.new_page(viewport={"width": 1200, "height": 5000})
            page.set_content(html_doc, wait_until="load")
            card = page.locator("#card")
            box = card.bounding_box()
            if box:
                print(f"CARD_BBOX: x={box['x']:.1f} y={box['y']:.1f} w={box['width']:.1f} h={box['height']:.1f}")
            quality = 88
            card.screenshot(path=output_path, type="jpeg", quality=quality, scale="device")
            while os.path.getsize(output_path) > 1_000_000 and quality > 45:
                quality -= 7
                card.screenshot(path=output_path, type="jpeg", quality=quality, scale="device")
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
