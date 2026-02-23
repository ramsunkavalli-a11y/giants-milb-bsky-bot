import argparse
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


def _normalize_name(name: str) -> str:
    base = unicodedata.normalize("NFKD", (name or "").strip().lower())
    return " ".join("".join(ch for ch in base if not unicodedata.combining(ch)).split())


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ip_to_outs(ip: str) -> int:
    if not ip:
        return 0
    parts = str(ip).split(".")
    whole = _safe_int(parts[0])
    frac = _safe_int(parts[1]) if len(parts) > 1 else 0
    frac = min(max(frac, 0), 2)
    return whole * 3 + frac


def _outs_to_ip(outs: int) -> str:
    return f"{outs // 3}.{outs % 3}"


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"bootstrapped": False, "seen_transaction_ids": [], "last_run_iso": None, "posted_games": {}, "recaps": {}}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("posted_games", {})
    state.setdefault("recaps", {})
    return state


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


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


def _derive_pitch_event_metrics(feed: Dict[str, Any]) -> Dict[int, Dict[str, int]]:
    metrics: Dict[int, Dict[str, int]] = {}
    plays = ((feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or []

    for play in plays:
        pitcher_id = _safe_int(((play.get("matchup") or {}).get("pitcher") or {}).get("id"), -1)
        if pitcher_id <= 0:
            continue
        m = metrics.setdefault(pitcher_id, {"swstr": 0, "gb": 0})

        for pe in play.get("playEvents") or []:
            if not pe.get("isPitch"):
                continue
            desc = ((pe.get("details") or {}).get("description") or "").lower()
            code = ((pe.get("details") or {}).get("code") or "").upper()
            if "swinging strike" in desc or code in {"S", "W", "Q", "T"}:
                m["swstr"] += 1
            trajectory = ((pe.get("hitData") or {}).get("trajectory") or "").lower()
            if "ground" in trajectory:
                m["gb"] += 1

    return metrics


def extract_player_lines(feed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    game_data = feed.get("gameData") or {}
    live_data = feed.get("liveData") or {}
    boxscore = live_data.get("boxscore") or {}
    teams = boxscore.get("teams") or {}

    orange_is_home = (((game_data.get("teams") or {}).get("home") or {}).get("id") == DSL_ORANGE_TEAM_ID)
    side_key = "home" if orange_is_home else "away"
    orange_team = teams.get(side_key) or {}
    player_map = orange_team.get("players") or {}

    batters_set = {_safe_int(x, -1) for x in (orange_team.get("batters") or [])}
    pitchers_set = {_safe_int(x, -1) for x in (orange_team.get("pitchers") or [])}
    pitch_metrics = _derive_pitch_event_metrics(feed)

    hitters: List[Dict[str, Any]] = []
    pitchers: List[Dict[str, Any]] = []

    for pdata in player_map.values():
        person = pdata.get("person") or {}
        pid = _safe_int(person.get("id"), -1)
        name = person.get("fullName", "Unknown")
        stats = pdata.get("stats") or {}
        batting = stats.get("batting") or {}
        pitching = stats.get("pitching") or {}
        batting_order = pdata.get("battingOrder")
        all_positions = pdata.get("allPositions") or []

        is_hitter_appearance = (
            pid in batters_set
            or bool(batting_order)
            or bool(all_positions)
            or any(k in batting for k in ("atBats", "plateAppearances", "hits", "runs"))
        )
        if is_hitter_appearance:
            hitters.append(
                {
                    "id": pid,
                    "name": name,
                    "batting_order": str(batting_order or ""),
                    "ab": _safe_int(batting.get("atBats")),
                    "h": _safe_int(batting.get("hits")),
                    "r": _safe_int(batting.get("runs")),
                    "bb": _safe_int(batting.get("baseOnBalls")),
                    "so": _safe_int(batting.get("strikeOuts")),
                    "hr": _safe_int(batting.get("homeRuns")),
                    "season_slash": _player_season_slash(pdata),
                }
            )

        is_pitcher_appearance = (
            pid in pitchers_set
            or bool(pitching)
            or any(k in pitching for k in ("inningsPitched", "strikeOuts", "numberOfPitches"))
        )
        if is_pitcher_appearance:
            p_event = pitch_metrics.get(pid, {})
            pitchers.append(
                {
                    "id": pid,
                    "name": name,
                    "ip": str(pitching.get("inningsPitched") or "0.0"),
                    "h": _safe_int(pitching.get("hits")),
                    "r": _safe_int(pitching.get("runs")),
                    "er": _safe_int(pitching.get("earnedRuns")),
                    "bb": _safe_int(pitching.get("baseOnBalls")),
                    "k": _safe_int(pitching.get("strikeOuts")),
                    "swstr": p_event.get("swstr"),
                    "gb": p_event.get("gb"),
                }
            )

    def _bat_order_key(h: Dict[str, Any]) -> Tuple[int, str]:
        order = h.get("batting_order") or ""
        try:
            return int(order), h["name"]
        except ValueError:
            return 9999, h["name"]

    hitters = sorted(hitters, key=_bat_order_key)
    pitchers = sorted(pitchers, key=lambda p: (-_ip_to_outs(p["ip"]), p["name"]))

    return hitters, pitchers, {"orange_is_home": orange_is_home, "side_key": side_key}


def _load_prospects() -> List[Dict[str, Any]]:
    if not PROSPECTS_PATH.exists():
        return []
    with open(PROSPECTS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("prospects", [])


def _prospect_priority_map() -> Dict[str, int]:
    priorities: Dict[str, int] = {}
    for p in _load_prospects():
        name = _normalize_name(p.get("name", ""))
        if not name:
            continue
        priorities[name] = _safe_int(p.get("priority"), 9)
    return priorities


def maybe_warn_stale_prospects() -> None:
    if not PROSPECTS_PATH.exists():
        print("WARNING: prospects.json not found")
        return
    mtime = datetime.fromtimestamp(PROSPECTS_PATH.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    if age_days > PROSPECTS_STALE_DAYS:
        print(f"WARNING: prospects.json is {age_days} days old; consider refreshing rankings.")


def _hitter_perf_score(h: Dict[str, Any]) -> float:
    return h["h"] * 2.5 + h["hr"] * 2 + h["bb"] + h["r"] - h["so"] * 0.2


def _pitcher_perf_score(p: Dict[str, Any]) -> float:
    return p["k"] * 1.7 + (_ip_to_outs(p["ip"]) / 3.0) - p["er"] * 2 - p["bb"] * 0.8


def _format_hitter_for_text(h: Dict[str, Any]) -> str:
    base = f"{h['name']} {h['h']}-{h['ab']}"
    if h.get("season_slash"):
        base += f" | {h['season_slash']}"
    return base


def _format_pitcher_for_text(p: Dict[str, Any]) -> str:
    metrics = [f"{p['ip']} IP", f"{p['k']} K"]
    if p.get("swstr"):
        metrics.append(f"{p['swstr']} SwStr")
    if p.get("gb"):
        metrics.append(f"{p['gb']} GB")
    return f"{p['name']} " + "/".join(metrics)


def select_post_entities(
    hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]], max_prospects: int = 2, max_notables: int = 2
) -> Tuple[List[str], List[str]]:
    priorities = _prospect_priority_map()

    entities: List[Tuple[float, str, str, bool, int]] = []
    for h in hitters:
        norm = _normalize_name(h["name"])
        is_prospect = norm in priorities
        p_priority = priorities.get(norm, 99)
        entities.append((_hitter_perf_score(h), _format_hitter_for_text(h), norm, is_prospect, p_priority))

    for p in pitchers:
        norm = _normalize_name(p["name"])
        is_prospect = norm in priorities
        p_priority = priorities.get(norm, 99)
        entities.append((_pitcher_perf_score(p), _format_pitcher_for_text(p), norm, is_prospect, p_priority))

    prospects = sorted([e for e in entities if e[3]], key=lambda e: (e[4], -e[0], e[1]))
    non_prospects = sorted([e for e in entities if not e[3]], key=lambda e: (-e[0], e[1]))

    chosen_prospects = [e[1] for e in prospects[:max_prospects]]
    used_norms = {e[2] for e in prospects[:max_prospects]}
    chosen_notables: List[str] = []
    for e in non_prospects:
        if e[2] in used_norms:
            continue
        chosen_notables.append(e[1])
        if len(chosen_notables) >= max_notables:
            break

    return chosen_prospects, chosen_notables


def _build_text_with_limits(line1: str, prospects: List[str], notables: List[str]) -> str:
    p_work = prospects[:]
    n_work = notables[:]

    while True:
        line2 = f"Prospects: {', '.join(p_work)}" if p_work else "Prospects: n/a"
        line3 = f"Notables: {', '.join(n_work)}" if n_work else "Notables: n/a"
        text = f"{line1}\n{line2}\n{line3}"
        if len(text) <= MAX_POST_CHARS:
            return text
        if n_work:
            n_work.pop()
            continue
        if p_work:
            p_work.pop()
            continue
        return (line1 + "\nProspects: n/a\nNotables: n/a")[:MAX_POST_CHARS]


def compose_post_text(status: str, orange_runs: int, opp_name: str, opp_runs: int, hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> str:
    line1 = f"{status}: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}"
    prospects, notables = select_post_entities(hitters, pitchers)
    return _build_text_with_limits(line1, prospects, notables)


def _shorten_play_desc(play: Dict[str, Any], orange_delta: int) -> str:
    result = play.get("result") or {}
    about = play.get("about") or {}
    matchup = play.get("matchup") or {}
    batter = ((matchup.get("batter") or {}).get("fullName") or "Player").split()[-1]
    event = (result.get("event") or "play").lower()
    runs_scored = _safe_int(result.get("rbi"), abs(orange_delta))
    half = "T" if (about.get("halfInning") or "").lower() == "top" else "B"
    inning = _safe_int(about.get("inning"), 1)
    run_text = f"{runs_scored}-run " if runs_scored > 1 else ""
    return f"{half}{inning}: {batter} {run_text}{event} (ORANGE {orange_delta:+d})"


def generate_key_moments(feed: Dict[str, Any], orange_is_home: bool) -> List[str]:
    plays = ((feed.get("liveData") or {}).get("plays") or {})
    all_plays = plays.get("allPlays") or []
    scoring_indices = set(plays.get("scoringPlays") or [])

    if not all_plays:
        return ["T1: No major scoring moments", "T1: No major scoring moments", "T1: No major scoring moments"]

    ranked: List[Tuple[float, str]] = []
    prev_home = 0
    prev_away = 0

    for idx, play in enumerate(all_plays):
        about = play.get("about") or {}
        result = play.get("result") or {}
        is_scoring = bool(about.get("isScoringPlay")) or idx in scoring_indices

        home_after = _safe_int(result.get("homeScore"), prev_home)
        away_after = _safe_int(result.get("awayScore"), prev_away)

        if is_scoring:
            orange_before = prev_home if orange_is_home else prev_away
            opp_before = prev_away if orange_is_home else prev_home
            orange_after = home_after if orange_is_home else away_after
            opp_after = away_after if orange_is_home else home_after

            diff_before = orange_before - opp_before
            diff_after = orange_after - opp_after
            orange_delta = orange_after - orange_before

            swing = abs(diff_after) - abs(diff_before)
            if diff_before == 0 and diff_after != 0:
                swing += 2.5
            if (diff_before < 0 <= diff_after) or (diff_before > 0 >= diff_after):
                swing += 3.0

            inning = _safe_int(about.get("inning"), 1)
            inning_weight = min(2.5, 1 + 0.15 * (inning - 1))
            score_weight = max(1.0, abs(orange_delta))
            proxy_importance = swing * inning_weight * score_weight

            ranked.append((proxy_importance, _shorten_play_desc(play, orange_delta)))

        prev_home, prev_away = home_after, away_after

    ranked.sort(key=lambda x: x[0], reverse=True)
    moments = [r[1] for r in ranked[:3]]
    while len(moments) < 3:
        moments.append("T1: No additional high-leverage scoring play")
    return moments


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_rule(draw: Any, y: int, width: int) -> None:
    draw.line([(40, y), (width - 40, y)], fill=(224, 224, 224), width=2)


def _draw_table(
    draw: Any,
    start_y: int,
    width: int,
    title: str,
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[int],
    row_h: int,
) -> int:
    title_font = _load_font(27)
    head_font = _load_font(19)
    row_font = _load_font(18)

    x0 = 40
    y = start_y
    draw.text((x0, y), title, fill=(0, 0, 0), font=title_font)
    y += 36

    draw.rectangle([x0, y, width - 40, y + row_h], fill=(245, 245, 245))
    x = x0 + 12
    for i, h in enumerate(headers):
        draw.text((x, y + 8), h, fill=(40, 40, 40), font=head_font)
        x += col_widths[i]
    y += row_h

    for idx, row in enumerate(rows):
        if idx % 2 == 1:
            draw.rectangle([x0, y, width - 40, y + row_h], fill=(250, 250, 250))
        x = x0 + 12
        for i, cell in enumerate(row):
            draw.text((x, y + 7), str(cell), fill=(20, 20, 20), font=row_font)
            x += col_widths[i]
        y += row_h

    _draw_rule(draw, y + 4, width)
    return y + 18


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
    from PIL import Image, ImageDraw

    width = 1080
    base_h = 260
    hitters_h = 40 + 34 + 31 * max(1, len(hitters))
    pitchers_h = 40 + 34 + 31 * max(1, len(pitchers))
    moments_h = 190
    height = max(1350, min(2200, base_h + hitters_h + pitchers_h + moments_h))

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(42)
    meta_font = _load_font(24)
    body_font = _load_font(22)

    y = 42
    draw.text((40, y), matchup, fill=(0, 0, 0), font=title_font)
    y += 54
    draw.text((40, y), f"{game_date}  •  {status}", fill=(45, 45, 45), font=meta_font)
    y += 36
    draw.text((40, y), score_line, fill=(0, 0, 0), font=body_font)
    y += 32
    if linescore:
        draw.text((40, y), linescore, fill=(70, 70, 70), font=meta_font)
        y += 30
    _draw_rule(draw, y, width)
    y += 18

    hitter_headers = ["Player", "AB", "R", "H", "BB", "K", "HR", "Season"]
    hitter_widths = [330, 70, 60, 60, 70, 60, 70, 250]
    hitter_rows = [
        [h["name"], h["ab"], h["r"], h["h"], h["bb"], h["so"], h["hr"], h.get("season_slash") or "-"]
        for h in hitters
    ]
    y = _draw_table(draw, y, width, "Hitters (all appearances)", hitter_headers, hitter_rows, hitter_widths, 30)

    pitcher_headers = ["Pitcher", "IP", "H", "R", "ER", "BB", "K", "SwStr", "GB"]
    pitcher_widths = [280, 80, 50, 50, 60, 60, 50, 90, 80]
    pitcher_rows = [
        [
            p["name"],
            p["ip"],
            p["h"],
            p["r"],
            p["er"],
            p["bb"],
            p["k"],
            p.get("swstr") if p.get("swstr") else "-",
            p.get("gb") if p.get("gb") else "-",
        ]
        for p in pitchers
    ]
    y = _draw_table(draw, y, width, "Pitchers (all appearances)", pitcher_headers, pitcher_rows, pitcher_widths, 30)

    draw.text((40, y), "Key moments", fill=(0, 0, 0), font=_load_font(27))
    y += 40
    for idx, moment in enumerate(key_moments[:3], start=1):
        draw.text((55, y), f"{idx}. {moment}", fill=(35, 35, 35), font=meta_font)
        y += 35

    quality = 88
    img.save(output_path, format="JPEG", quality=quality, optimize=True)
    while os.path.getsize(output_path) > 1_000_000 and quality > 50:
        quality -= 6
        img.save(output_path, format="JPEG", quality=quality, optimize=True)

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
    if not games:
        return None

    dated_games: List[Tuple[str, int]] = []
    for g in games:
        gpk = g.get("gamePk")
        gdate = g.get("officialDate") or ""
        if gpk:
            dated_games.append((gdate, int(gpk)))
    if not dated_games:
        return None
    dated_games.sort(reverse=True)
    return dated_games[0][1]


def _get_game_from_overrides(session: requests.Session) -> List[int]:
    override_gamepk = (os.getenv("OVERRIDE_GAMEPK") or "").strip()
    override_date = (os.getenv("OVERRIDE_DATE") or "").strip()

    if override_gamepk:
        print(f"Using OVERRIDE_GAMEPK={override_gamepk}")
        return [_safe_int(override_gamepk)]

    if override_date:
        print(f"Using OVERRIDE_DATE={override_date}")
        games = fetch_schedule_games(session, target_date=override_date)
        return [int(g["gamePk"]) for g in games if g.get("gamePk")]

    if (os.getenv("GITHUB_EVENT_NAME") or "") == "workflow_dispatch":
        recent = _most_recent_gamepk_from_window(session)
        if recent:
            print(f"No override provided on workflow_dispatch; using most recent gamePk={recent}")
            return [recent]
        print("No override provided on workflow_dispatch; no recent DSL Orange game found")

    return []


def _orange_score_tuple(feed: Dict[str, Any], orange_is_home: bool) -> Tuple[int, int]:
    ls = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
    home_runs = _safe_int(((ls.get("home") or {}).get("runs") or 0))
    away_runs = _safe_int(((ls.get("away") or {}).get("runs") or 0))
    orange = home_runs if orange_is_home else away_runs
    opp = away_runs if orange_is_home else home_runs
    return orange, opp


def _should_post_game(state_entry: Dict[str, Any], status: str, force_repost: bool = False) -> bool:
    if force_repost:
        return True
    if status == "Suspended" and not state_entry.get("posted_suspended", False):
        return True
    if status == "Final" and not state_entry.get("posted_final", False):
        return True
    return False


def update_state(state: Dict[str, Any], game_pk: int, status: str, posted: bool) -> None:
    game_key = str(game_pk)
    entry = state.setdefault("posted_games", {}).setdefault(
        game_key,
        {"posted_suspended": False, "posted_final": False, "last_status": None, "last_seen_iso": None},
    )
    if posted:
        if status == "Suspended":
            entry["posted_suspended"] = True
        if status == "Final":
            entry["posted_final"] = True
    entry["last_status"] = status
    entry["last_seen_iso"] = datetime.now(timezone.utc).isoformat()


def run_finals_mode() -> None:
    state = load_state()
    session = make_session()
    maybe_warn_stale_prospects()

    game_pks = _get_game_from_overrides(session)
    if not game_pks:
        game_pks = [int(g["gamePk"]) for g in fetch_schedule_games(session) if g.get("gamePk")]

    if not game_pks:
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print("No DSL Orange games in schedule window.")
        return

    client: Optional[Client] = None
    posted_count = 0
    dry_run = os.getenv("DRY_RUN", "0") == "1"
    force_repost = os.getenv("FORCE_REPOST", "0") == "1"

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

        entry = state.setdefault("posted_games", {}).setdefault(
            str(game_pk),
            {"posted_suspended": False, "posted_final": False, "last_status": None, "last_seen_iso": None},
        )
        if not _should_post_game(entry, status, force_repost=force_repost):
            update_state(state, game_pk, status, posted=False)
            continue

        hitters, pitchers, _ = extract_player_lines(feed)
        key_moments = generate_key_moments(feed, orange_is_home=orange_is_home)

        orange_runs, opp_runs = _orange_score_tuple(feed, orange_is_home=orange_is_home)
        linescore_obj = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
        home_line = linescore_obj.get("home") or {}
        away_line = linescore_obj.get("away") or {}
        linescore = (
            f"R/H/E — {away.get('abbreviation', 'AWY')} {away_line.get('runs', '-')}/{away_line.get('hits', '-')}/{away_line.get('errors', '-')}"
            f" | {home.get('abbreviation', 'HME')} {home_line.get('runs', '-')}/{home_line.get('hits', '-')}/{home_line.get('errors', '-')}"
        )

        matchup = f"{away.get('name', 'Away')} at {home.get('name', 'Home')}"
        game_date = (game_data.get("datetime") or {}).get("officialDate") or datetime.now(timezone.utc).date().isoformat()
        score_line = f"DSL Giants Orange {orange_runs} - {opp_runs} {opp_name}"
        image_path = f"dsl_orange_{game_pk}_{status.lower()}.jpg"
        render_boxscore_card_image(
            output_path=image_path,
            matchup=matchup,
            game_date=game_date,
            status=status,
            score_line=score_line,
            linescore=linescore,
            hitters=hitters,
            pitchers=pitchers,
            key_moments=key_moments,
        )

        text = compose_post_text(status, orange_runs, opp_name, opp_runs, hitters, pitchers)
        alt_text = f"{status} card: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}."

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
    wins = 0
    losses = 0
    highlights: List[str] = []

    for g in games:
        feed = fetch_game_feed(session, int(g["gamePk"]))
        if not feed:
            continue
        status = classify_terminal_status(feed)
        if status != "Final":
            continue

        gd = feed.get("gameData") or {}
        teams = gd.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        orange_is_home = home.get("id") == DSL_ORANGE_TEAM_ID
        opp_name = away.get("name", "Opponent") if orange_is_home else home.get("name", "Opponent")

        orange_runs, opp_runs = _orange_score_tuple(feed, orange_is_home)
        if orange_runs > opp_runs:
            wins += 1
        elif orange_runs < opp_runs:
            losses += 1
        finals.append(f"Final: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}")

        hitters, pitchers, _ = extract_player_lines(feed)
        prospects, notables = select_post_entities(hitters, pitchers, max_prospects=1, max_notables=1)
        if prospects:
            highlights.append(prospects[0])
        elif notables:
            highlights.append(notables[0])

    if not finals:
        print(f"No final games to recap on {target_day}.")
        state["recaps"][target_day] = {"posted": False, "note": "no_finals"}
        save_state(state)
        return

    summary = f"Yesterday: DSL Orange went {wins}-{losses}. " + " | ".join(finals[:2])
    if highlights:
        summary += f" — Prospects/Notables: {', '.join(highlights[:2])}."
    summary = summary[:MAX_POST_CHARS]

    client = bsky_login()
    client.send_post(text=summary)
    time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    state["recaps"][target_day] = {"posted": True, "at": datetime.now(timezone.utc).isoformat()}
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"Posted recap for {target_day}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DSL Giants Orange gameday poster")
    parser.add_argument("--recap", action="store_true", help="Run daily recap mode")
    args = parser.parse_args()

    if args.recap:
        run_daily_recap_mode()
    else:
        run_finals_mode()


if __name__ == "__main__":
    main()
