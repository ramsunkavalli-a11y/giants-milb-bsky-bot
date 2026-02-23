import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from atproto import Client, models

from bot import STATE_PATH, SLEEP_BETWEEN_POSTS_SEC, make_session

API_BASE = "https://statsapi.mlb.com"
DSL_ORANGE_TEAM_ID = 615
SPORT_ID = 16
MAX_POST_CHARS = 300
PROSPECTS_PATH = Path("prospects.json")
PROSPECTS_STALE_DAYS = 45


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


def fetch_schedule_games(session: requests.Session, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"sportId": SPORT_ID, "teamId": DSL_ORANGE_TEAM_ID}
    if target_date:
        params["date"] = target_date
    else:
        params["startDate"] = (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()
        params["endDate"] = datetime.now(timezone.utc).date().isoformat()

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
    avg = season_bat.get("avg")
    obp = season_bat.get("obp")
    slg = season_bat.get("slg")
    if avg and obp and slg:
        return f"{avg}/{obp}/{slg}"
    return None


def extract_player_lines(feed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    game_data = feed.get("gameData") or {}
    live_data = feed.get("liveData") or {}
    boxscore = live_data.get("boxscore") or {}
    teams = (boxscore.get("teams") or {})

    orange_is_home = (((game_data.get("teams") or {}).get("home") or {}).get("id") == DSL_ORANGE_TEAM_ID)
    side_key = "home" if orange_is_home else "away"
    orange_team = teams.get(side_key) or {}
    player_map = orange_team.get("players") or {}

    hitters: List[Dict[str, Any]] = []
    pitchers: List[Dict[str, Any]] = []

    for pdata in player_map.values():
        person = pdata.get("person") or {}
        name = person.get("fullName", "Unknown")
        batting = ((pdata.get("stats") or {}).get("batting") or {})
        pitching = ((pdata.get("stats") or {}).get("pitching") or {})

        if batting and any(str(v) not in {"0", "0.0", ""} for v in batting.values()):
            hitters.append(
                {
                    "name": name,
                    "ab": batting.get("atBats", 0),
                    "h": batting.get("hits", 0),
                    "hr": batting.get("homeRuns", 0),
                    "bb": batting.get("baseOnBalls", 0),
                    "so": batting.get("strikeOuts", 0),
                    "r": batting.get("runs", 0),
                    "season_slash": _player_season_slash(pdata),
                }
            )

        if pitching and any(str(v) not in {"0", "0.0", ""} for v in pitching.values()):
            pitchers.append(
                {
                    "name": name,
                    "ip": pitching.get("inningsPitched", "0.0"),
                    "h": pitching.get("hits", 0),
                    "r": pitching.get("runs", 0),
                    "er": pitching.get("earnedRuns", 0),
                    "bb": pitching.get("baseOnBalls", 0),
                    "k": pitching.get("strikeOuts", 0),
                    "pitches": pitching.get("numberOfPitches", 0),
                }
            )

    return hitters, pitchers, {"orange_is_home": orange_is_home, "side_key": side_key}


def _load_prospects() -> List[Dict[str, Any]]:
    if not PROSPECTS_PATH.exists():
        return []
    with open(PROSPECTS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("prospects", [])


def maybe_warn_stale_prospects() -> None:
    if not PROSPECTS_PATH.exists():
        print("WARNING: prospects.json not found")
        return
    mtime = datetime.fromtimestamp(PROSPECTS_PATH.stat().st_mtime, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - mtime).days
    if age_days > PROSPECTS_STALE_DAYS:
        print(f"WARNING: prospects.json is {age_days} days old; consider refreshing rankings.")


def select_key_hitters_pitchers(hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prospects = _load_prospects()
    prospect_names = {p.get("name", "").lower(): p.get("priority", 9) for p in prospects}

    def hitter_score(h: Dict[str, Any]) -> float:
        base = h["h"] * 2 + h["hr"] * 3 + h["bb"] + h["r"]
        bonus = 4 if h["name"].lower() in prospect_names else 0
        return base + bonus

    def pitcher_score(p: Dict[str, Any]) -> float:
        ip = float(str(p["ip"]).replace(".", "", 1)) if str(p["ip"]).replace(".", "", 1).isdigit() else 0.0
        base = p["k"] * 1.5 + ip - p["er"] * 2 - p["bb"] * 0.8
        bonus = 4 if p["name"].lower() in prospect_names else 0
        return base + bonus

    hitters_sorted = sorted(hitters, key=hitter_score, reverse=True)
    pitchers_sorted = sorted(pitchers, key=pitcher_score, reverse=True)
    return hitters_sorted[:4], pitchers_sorted[:3]


def _orange_score_tuple(feed: Dict[str, Any], orange_is_home: bool) -> Tuple[int, int]:
    ls = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
    home_runs = int(((ls.get("home") or {}).get("runs") or 0))
    away_runs = int(((ls.get("away") or {}).get("runs") or 0))
    orange = home_runs if orange_is_home else away_runs
    opp = away_runs if orange_is_home else home_runs
    return orange, opp


def generate_key_plays(feed: Dict[str, Any], orange_is_home: bool) -> List[str]:
    plays = ((feed.get("liveData") or {}).get("plays") or {})
    all_plays = plays.get("allPlays") or []
    scoring_indices = set(plays.get("scoringPlays") or [])
    if not all_plays:
        return []

    scored_candidates = []
    prev_home = 0
    prev_away = 0

    for idx, play in enumerate(all_plays):
        result = play.get("result") or {}
        about = play.get("about") or {}
        is_scoring = bool(about.get("isScoringPlay")) or idx in scoring_indices
        home_after = int(result.get("homeScore", prev_home))
        away_after = int(result.get("awayScore", prev_away))

        if is_scoring:
            orange_before = prev_home if orange_is_home else prev_away
            opp_before = prev_away if orange_is_home else prev_home
            orange_after = home_after if orange_is_home else away_after
            opp_after = away_after if orange_is_home else home_after
            diff_before = orange_before - opp_before
            diff_after = orange_after - opp_after

            swing = abs(diff_after) - abs(diff_before)
            if (diff_before <= 0 < diff_after) or (diff_before >= 0 > diff_after) or (diff_before == 0 and diff_after != 0):
                swing += 2.5
            inning = int(about.get("inning") or 1)
            inning_weight = min(2.5, 1 + 0.15 * (inning - 1))
            proxy = swing * inning_weight

            desc = result.get("description") or "Scoring play"
            half = "T" if about.get("halfInning") == "top" else "B"
            delta = diff_after - diff_before
            delta_txt = f"ORANGE {delta:+d}"
            scored_candidates.append(
                {
                    "proxy": proxy,
                    "text": f"{half}{inning}: {desc} ({delta_txt})",
                }
            )

        prev_home = home_after
        prev_away = away_after

    scored_candidates.sort(key=lambda x: x["proxy"], reverse=True)
    return [c["text"] for c in scored_candidates[:3]]




def _draw_line(draw: Any, y: int, width: int) -> None:
    draw.line([(60, y), (width - 60, y)], fill=(215, 215, 215), width=2)


def render_boxscore_image(
    output_path: str,
    matchup: str,
    game_date: str,
    status: str,
    score_line: str,
    linescore: str,
    hitters: List[Dict[str, Any]],
    pitchers: List[Dict[str, Any]],
    key_plays: List[str],
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    def _load_font(size: int) -> Any:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    width, height = 1080, 1350
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(42)
    body_font = _load_font(30)
    small_font = _load_font(24)

    y = 60
    draw.text((60, y), matchup, fill=(0, 0, 0), font=title_font)
    y += 45
    draw.text((60, y), f"{game_date} • {status}", fill=(35, 35, 35), font=body_font)
    y += 35
    draw.text((60, y), score_line, fill=(0, 0, 0), font=body_font)
    y += 35
    if linescore:
        draw.text((60, y), linescore, fill=(70, 70, 70), font=small_font)
        y += 30
    _draw_line(draw, y, width)

    y += 24
    draw.text((60, y), "Key Hitters", fill=(0, 0, 0), font=body_font)
    y += 34
    for h in hitters:
        slash = f" | {h['season_slash']}" if h.get("season_slash") else ""
        row = f"{h['name']}: {h['h']}-{h['ab']}, R {h['r']}, HR {h['hr']}, BB {h['bb']}, K {h['so']}{slash}"
        draw.text((60, y), row[:110], fill=(15, 15, 15), font=small_font)
        y += 28

    y += 8
    _draw_line(draw, y, width)
    y += 24
    draw.text((60, y), "Key Pitchers", fill=(0, 0, 0), font=body_font)
    y += 34
    for p in pitchers:
        row = f"{p['name']}: IP {p['ip']} H {p['h']} R {p['r']} ER {p['er']} BB {p['bb']} K {p['k']}"
        draw.text((60, y), row[:110], fill=(15, 15, 15), font=small_font)
        y += 28

    if key_plays:
        y += 8
        _draw_line(draw, y, width)
        y += 24
        draw.text((60, y), "Key Plays", fill=(0, 0, 0), font=body_font)
        y += 32
        for kp in key_plays[:3]:
            draw.text((60, y), f"• {kp}"[:115], fill=(40, 40, 40), font=small_font)
            y += 26

    quality = 87
    img.save(output_path, format="JPEG", quality=quality, optimize=True)
    while os.path.getsize(output_path) > 1_000_000 and quality > 55:
        quality -= 5
        img.save(output_path, format="JPEG", quality=quality, optimize=True)

    if os.path.getsize(output_path) > 1_000_000:
        img.save(output_path, format="PNG", optimize=True)

    return output_path


def _compose_text(status: str, orange_runs: int, opp_name: str, opp_runs: int, hitters: List[Dict[str, Any]], pitchers: List[Dict[str, Any]]) -> str:
    prospect_mentions = [h for h in hitters if h.get("season_slash")]
    top_h = hitters[0] if hitters else None
    top_p = pitchers[0] if pitchers else None

    prospects_txt = ""
    if prospect_mentions:
        p = prospect_mentions[0]
        prospects_txt = f"Prospects: {p['name']} {p['h']}-{p['ab']}"
    elif top_h:
        prospects_txt = f"Prospects: {top_h['name']} {top_h['h']}-{top_h['ab']}"

    notable_bits = []
    if top_p:
        notable_bits.append(f"{top_p['name']} {top_p['ip']} IP/{top_p['k']} K")
    if top_h and (not prospect_mentions or top_h['name'] != prospect_mentions[0]['name']):
        notable_bits.append(f"{top_h['name']} {top_h['h']}-{top_h['ab']}")
    notables = "; ".join(notable_bits[:2])

    text = f"{status}: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs} — {prospects_txt}"
    if notables:
        text += f". Notables: {notables}."
    return text[:MAX_POST_CHARS]


def post_to_bluesky_with_image(client: Client, text: str, image_path: str, alt_text: str) -> None:
    with open(image_path, "rb") as f:
        blob_resp = client.upload_blob(f.read())

    image_blob = blob_resp.get("blob") if isinstance(blob_resp, dict) else getattr(blob_resp, "blob", blob_resp)
    embed = models.AppBskyEmbedImages.Main(images=[models.AppBskyEmbedImages.Image(alt=alt_text, image=image_blob)])
    client.send_post(text=text, embed=embed)
    time.sleep(SLEEP_BETWEEN_POSTS_SEC)


def _get_game_from_overrides(session: requests.Session) -> List[int]:
    override_gamepk = os.getenv("OVERRIDE_GAMEPK")
    override_date = os.getenv("OVERRIDE_DATE")
    if override_gamepk:
        return [int(override_gamepk)]
    if override_date:
        games = fetch_schedule_games(session, target_date=override_date)
        return [int(g["gamePk"]) for g in games if g.get("gamePk")]
    return []


def _should_post_game(state_entry: Dict[str, Any], status: str) -> bool:
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
        if not _should_post_game(entry, status):
            update_state(state, game_pk, status, posted=False)
            continue

        hitters, pitchers, _ = extract_player_lines(feed)
        key_hitters, key_pitchers = select_key_hitters_pitchers(hitters, pitchers)
        key_plays = generate_key_plays(feed, orange_is_home=orange_is_home)

        orange_runs, opp_runs = _orange_score_tuple(feed, orange_is_home=orange_is_home)
        linescore_obj = ((feed.get("liveData") or {}).get("linescore") or {}).get("teams") or {}
        home_line = linescore_obj.get("home") or {}
        away_line = linescore_obj.get("away") or {}
        linescore = f"R/H/E — {away.get('abbreviation', 'AWY')} {away_line.get('runs', '-')}/{away_line.get('hits', '-')}/{away_line.get('errors', '-')} | {home.get('abbreviation', 'HME')} {home_line.get('runs', '-')}/{home_line.get('hits', '-')}/{home_line.get('errors', '-')}"

        matchup = f"{away.get('name', 'Away')} at {home.get('name', 'Home')}"
        game_date = (game_data.get("datetime") or {}).get("officialDate") or datetime.now(timezone.utc).date().isoformat()
        score_line = f"DSL Giants Orange {orange_runs} - {opp_runs} {opp_name}"
        image_path = f"dsl_orange_{game_pk}_{status.lower()}.jpg"
        render_boxscore_image(
            output_path=image_path,
            matchup=matchup,
            game_date=game_date,
            status=status,
            score_line=score_line,
            linescore=linescore,
            hitters=key_hitters,
            pitchers=key_pitchers,
            key_plays=key_plays,
        )

        text = _compose_text(status, orange_runs, opp_name, opp_runs, key_hitters, key_pitchers)
        alt_text = f"{status} box score card: DSL Giants Orange {orange_runs}, {opp_name} {opp_runs}."

        if dry_run:
            print(f"DRY_RUN: would post {status} for game {game_pk}: {text}")
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

    games = fetch_schedule_games(session, target_day)
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
        key_hitters, key_pitchers = select_key_hitters_pitchers(hitters, pitchers)
        if key_hitters:
            highlights.append(f"{key_hitters[0]['name']} {key_hitters[0]['h']}-{key_hitters[0]['ab']}")
        elif key_pitchers:
            highlights.append(f"{key_pitchers[0]['name']} {key_pitchers[0]['ip']} IP/{key_pitchers[0]['k']} K")

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
