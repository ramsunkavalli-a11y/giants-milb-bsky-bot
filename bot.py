import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from atproto import Client

# -----------------------------
# Config
# -----------------------------
MAX_CHARS = 300
LOOKBACK_DAYS = 14
RECENT_DAYS = 14
MIN_RECENT_HITTER_PA = 20
MIN_RECENT_PITCHER_IP = 5.0
SLEEP_BETWEEN_POSTS_SEC = 1.2
RECENT_EVENT_AUDIT_LIMIT = 200

# STATE_PATH is retained because the disabled DSL code imports it.
STATE_PATH = "state.json"
TRANSACTION_STATE_PATH = "transaction_state.json"
API_BASE = "https://statsapi.mlb.com/api/v1"

SF = 137
DSL_BLACK = 2134
DSL_ORANGE = 615
ACL = 408
SAN_JOSE = 476
EUGENE = 461
RICHMOND = 3410
SACRAMENTO = 105

TRACKED_TEAM_IDS: Set[int] = {DSL_BLACK, DSL_ORANGE, ACL, SAN_JOSE, EUGENE, RICHMOND, SACRAMENTO}
ORG_TEAM_IDS: Set[int] = TRACKED_TEAM_IDS | {SF}

TEAM_HEADER: Dict[int, str] = {
    DSL_BLACK: "DSL Giants",
    DSL_ORANGE: "DSL Giants",
    ACL: "ACL Giants",
    SAN_JOSE: "San Jose",
    EUGENE: "Eugene",
    RICHMOND: "Richmond",
    SACRAMENTO: "Sacramento",
}

TEAM_SHORT: Dict[int, str] = {
    DSL_BLACK: "DSL",
    DSL_ORANGE: "DSL",
    ACL: "ACL",
    SAN_JOSE: "San Jose",
    EUGENE: "Eugene",
    RICHMOND: "Richmond",
    SACRAMENTO: "Sacramento",
    SF: "SF",
}

TEAM_DESTINATION: Dict[int, str] = {
    DSL_BLACK: "DSL Giants",
    DSL_ORANGE: "DSL Giants",
    ACL: "ACL Giants",
    SAN_JOSE: "Low-A San Jose",
    EUGENE: "High-A Eugene",
    RICHMOND: "Double-A Richmond",
    SACRAMENTO: "Triple-A Sacramento",
    SF: "San Francisco",
}

TEAM_STAT_LABEL: Dict[int, str] = {
    DSL_BLACK: "DSL",
    DSL_ORANGE: "DSL",
    ACL: "ACL",
    SAN_JOSE: "SJ",
    EUGENE: "EUG",
    RICHMOND: "RIC",
    SACRAMENTO: "SAC",
    SF: "SF",
}

SECTION_ORDER = ["DSL Giants", "ACL Giants", "San Jose", "Eugene", "Richmond", "Sacramento"]

POSITION_RE = re.compile(r"\b(LHP|RHP|P|C|1B|2B|3B|SS|LF|CF|RF|OF|IF|INF|DH)\b", re.IGNORECASE)


# -----------------------------
# HTTP helpers
# -----------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "giants-milb-transactions-bot/2.0"})
    return s


# -----------------------------
# State
# -----------------------------
def _read_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(path):
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def load_transaction_state() -> Dict[str, Any]:
    """Load transaction-only state, migrating the legacy seen-ID list if needed."""
    if os.path.exists(TRANSACTION_STATE_PATH):
        state = _read_json(TRANSACTION_STATE_PATH, {})
    else:
        legacy = _read_json(STATE_PATH, {})
        state = {
            "bootstrapped": bool(legacy.get("bootstrapped", False)),
            "seen_transaction_ids": legacy.get("seen_transaction_ids", []),
            "seen_event_keys": [],
            "recent_events": [],
            "migrated_from_legacy_state": bool(legacy),
        }

    state.setdefault("bootstrapped", False)
    state.setdefault("seen_transaction_ids", [])
    state.setdefault("seen_event_keys", [])
    state.setdefault("recent_events", [])
    return state


def save_transaction_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["recent_events"] = state.get("recent_events", [])[-RECENT_EVENT_AUDIT_LIMIT:]
    _write_json(TRANSACTION_STATE_PATH, state)


# -----------------------------
# Data model
# -----------------------------
@dataclass
class TxnEvent:
    id: int
    sort_date: str
    person_id: Optional[int]
    person_name: str
    position: str
    event_type: str
    from_id: Optional[int]
    from_name: Optional[str]
    to_id: Optional[int]
    to_name: Optional[str]
    display_team_id: int
    header: str
    description: str
    event_key: str
    stats_lines: List[str] = field(default_factory=list)

    @property
    def is_level_change(self) -> bool:
        return (
            self.from_id in ORG_TEAM_IDS
            and self.to_id in ORG_TEAM_IDS
            and self.from_id != self.to_id
            and self.event_type in {"assignment", "optioned", "recalled"}
        )


@dataclass
class PostBundle:
    text: str
    event_ids: List[int]


# -----------------------------
# Transaction fetch / normalize
# -----------------------------
def fetch_transactions(s: requests.Session, team_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    r = s.get(
        f"{API_BASE}/transactions",
        params={"teamId": str(team_id), "startDate": start_date.isoformat(), "endDate": end_date.isoformat()},
        timeout=30,
    )
    r.raise_for_status()
    return (r.json() or {}).get("transactions", [])


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def pick_sort_date(t: Dict[str, Any]) -> str:
    return t.get("effectiveDate") or t.get("date") or t.get("resolutionDate") or ""


def parse_event_date(value: str) -> date:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return datetime.now(ZoneInfo("America/Los_Angeles")).date()


def get_team_fields(t: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    f = t.get("fromTeam") or {}
    to = t.get("toTeam") or {}
    return f.get("id"), f.get("name"), to.get("id"), to.get("name")


def clean_team_name(team_name: Optional[str]) -> str:
    n = normalize(team_name or "")
    for suffix in ("Flying Squirrels", "River Cats", "Emeralds"):
        n = re.sub(rf"\b{re.escape(suffix)}\b", "", n, flags=re.IGNORECASE).strip()
    n = re.sub(r"\bGiants\b", "", n, flags=re.IGNORECASE).strip()
    return n


def short_team(team_id: Optional[int], team_name: Optional[str]) -> str:
    if team_id is not None and int(team_id) in TEAM_SHORT:
        return TEAM_SHORT[int(team_id)]
    return clean_team_name(team_name) or "?"


def destination_name(team_id: Optional[int], team_name: Optional[str]) -> str:
    if team_id is not None and int(team_id) in TEAM_DESTINATION:
        return TEAM_DESTINATION[int(team_id)]
    return normalize(team_name or "") or "?"


def is_internal_dsl_move(from_id: Optional[int], to_id: Optional[int], desc_lower: str) -> bool:
    if (from_id in {DSL_BLACK, DSL_ORANGE}) and (to_id in {DSL_BLACK, DSL_ORANGE}) and (from_id != to_id):
        return True
    return "dsl giants black" in desc_lower and "dsl giants orange" in desc_lower


def choose_display_team_id(from_id: Optional[int], to_id: Optional[int], query_team_id: int) -> Optional[int]:
    if to_id in TRACKED_TEAM_IDS:
        return int(to_id)
    if from_id in TRACKED_TEAM_IDS:
        return int(from_id)
    if query_team_id in TRACKED_TEAM_IDS:
        return int(query_team_id)
    return None


def classify_event(desc: str) -> str:
    d = normalize(desc).lower()
    if "selected the contract" in d or "contract selected" in d:
        return "selected"
    if "recalled" in d:
        return "recalled"
    if "optioned" in d:
        return "optioned"
    if "rehab assignment" in d:
        return "rehab"
    if ("placed" in d and "injured list" in d) or ("placed" in d and re.search(r"\b\d+-day il\b", d)):
        return "il_placed"
    if ("activated" in d or "reinstated" in d) and ("injured list" in d or " il" in d):
        return "il_activated"
    if re.search(r"\breleased\b", d):
        return "released"
    if re.search(r"\bsigned\b", d):
        return "signed"
    if re.search(r"\bassigned\b", d):
        return "assignment"
    return "other"


def extract_position(desc: str, full_name: str = "") -> str:
    d = normalize(desc)
    if full_name:
        m = re.search(rf"\b(LHP|RHP|P|C|1B|2B|3B|SS|LF|CF|RF|OF|IF|INF|DH)\s+{re.escape(full_name)}\b", d, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = POSITION_RE.search(d)
    return m.group(1).upper() if m else ""


def canonical_description(desc: str) -> str:
    d = normalize(desc).lower()
    d = re.sub(r"\bthe\b", "", d)
    d = re.sub(r"\s+", " ", d).strip(" .")
    return d


def make_event_key(
    sort_date: str,
    person_id: Optional[int],
    event_type: str,
    from_id: Optional[int],
    to_id: Optional[int],
    desc: str,
) -> str:
    return "|".join(
        [
            (sort_date or "")[:10],
            str(person_id or 0),
            event_type,
            str(from_id or 0),
            str(to_id or 0),
            canonical_description(desc),
        ]
    )


def replace_injured_list_with_il(desc: str) -> str:
    return normalize(re.sub(r"\binjured list\b", "IL", desc, flags=re.IGNORECASE))


def strip_affiliate_prefix(desc: str, header: str, from_name: Optional[str], to_name: Optional[str]) -> str:
    d = normalize(desc)
    candidates = [
        to_name or "",
        from_name or "",
        "Richmond Flying Squirrels",
        "Sacramento River Cats",
        "Eugene Emeralds",
        "San Jose Giants",
        "ACL Giants",
        "DSL Giants Orange",
        "DSL Giants Black",
        header,
    ]
    for prefix in sorted({normalize(x) for x in candidates if x}, key=len, reverse=True):
        if d.lower().startswith(prefix.lower()):
            return d[len(prefix):].lstrip(" :-")
    return d


def make_compact_line(event: TxnEvent) -> str:
    person = " ".join(x for x in [event.position, event.person_name] if x).strip() or event.person_name

    if event.event_type == "assignment" and event.to_id in ORG_TEAM_IDS:
        origin = destination_name(event.from_id, event.from_name) if event.from_id else ""
        if origin and origin != "?":
            return f"{person} → {destination_name(event.to_id, event.to_name)} from {origin}."
        return f"{person} → {destination_name(event.to_id, event.to_name)}."

    d = replace_injured_list_with_il(event.description)
    d = strip_affiliate_prefix(d, event.header, event.from_name, event.to_name)
    return normalize(d)


# -----------------------------
# Player / stats context
# -----------------------------
def fetch_person_position(s: requests.Session, person_id: int) -> str:
    try:
        r = s.get(f"{API_BASE}/people/{person_id}", timeout=30)
        r.raise_for_status()
        person = ((r.json() or {}).get("people") or [{}])[0]
        return ((person.get("primaryPosition") or {}).get("abbreviation") or "").upper()
    except (requests.RequestException, IndexError, TypeError):
        return ""


def _stats_split(payload: Dict[str, Any]) -> Dict[str, Any]:
    for group in payload.get("stats") or []:
        splits = group.get("splits") or []
        if splits:
            return splits[0].get("stat") or {}
    return {}


def fetch_date_range_stats(
    s: requests.Session,
    person_id: int,
    group: str,
    team_id: int,
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    if end_date < start_date:
        return {}
    try:
        r = s.get(
            f"{API_BASE}/people/{person_id}/stats",
            params={
                "stats": "byDateRange",
                "group": group,
                "startDate": start_date.isoformat(),
                "endDate": end_date.isoformat(),
                "teamId": str(team_id),
            },
            timeout=30,
        )
        r.raise_for_status()
        return _stats_split(r.json() or {})
    except requests.RequestException as exc:
        print(f"WARNING: stats fetch failed for {person_id} ({group}, team {team_id}): {exc}")
        return {}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_rate(value: Any) -> str:
    s = str(value or "")
    if s.startswith("0."):
        return s[1:]
    return s


def _pitching_rates(stat: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    bf = _to_float(stat.get("battersFaced"))
    if bf <= 0:
        return None, None
    k_pct = (_to_float(stat.get("strikeOuts")) / bf) * 100
    bb_pct = (_to_float(stat.get("baseOnBalls")) / bf) * 100
    return k_pct, bb_pct


def format_hitter_stats(stat: Dict[str, Any], include_power: bool = True) -> str:
    pa = int(_to_float(stat.get("plateAppearances")))
    avg, obp, slg = stat.get("avg"), stat.get("obp"), stat.get("slg")
    if pa <= 0 or not (avg and obp and slg):
        return ""
    out = f"{_fmt_rate(avg)}/{_fmt_rate(obp)}/{_fmt_rate(slg)} in {pa} PA"
    hr = int(_to_float(stat.get("homeRuns")))
    if include_power and hr > 0:
        out += f", {hr} HR"
    return out


def format_pitcher_stats(stat: Dict[str, Any]) -> str:
    ip = str(stat.get("inningsPitched") or "")
    era = str(stat.get("era") or "")
    if not ip or not era:
        return ""
    out = f"{ip} IP, {era} ERA"
    k_pct, bb_pct = _pitching_rates(stat)
    if k_pct is not None and bb_pct is not None:
        out += f" · {k_pct:.1f}% K, {bb_pct:.1f}% BB"
    return out


def innings_as_float(stat: Dict[str, Any]) -> float:
    raw = str(stat.get("inningsPitched") or "0.0")
    try:
        whole, frac = (raw.split(".", 1) + ["0"])[:2]
        outs = int(whole) * 3 + min(max(int(frac[:1] or 0), 0), 2)
        return outs / 3.0
    except ValueError:
        return 0.0


def attach_stats_context(s: requests.Session, event: TxnEvent) -> None:
    if not event.is_level_change or not event.person_id or event.from_id not in ORG_TEAM_IDS:
        return

    pos = event.position or fetch_person_position(s, event.person_id)
    event.position = pos
    is_pitcher = pos in {"P", "RHP", "LHP"}
    group = "pitching" if is_pitcher else "hitting"

    move_date = parse_event_date(event.sort_date)
    end = move_date - timedelta(days=1)
    season_start = date(move_date.year, 1, 1)
    recent_start = move_date - timedelta(days=RECENT_DAYS)

    season = fetch_date_range_stats(s, event.person_id, group, int(event.from_id), season_start, end)
    recent = fetch_date_range_stats(s, event.person_id, group, int(event.from_id), recent_start, end)

    label = TEAM_STAT_LABEL.get(int(event.from_id), short_team(event.from_id, event.from_name))

    if is_pitcher:
        season_text = format_pitcher_stats(season)
        recent_text = format_pitcher_stats(recent) if innings_as_float(recent) >= MIN_RECENT_PITCHER_IP else ""
    else:
        season_text = format_hitter_stats(season, include_power=True)
        recent_pa = int(_to_float(recent.get("plateAppearances")))
        recent_text = format_hitter_stats(recent, include_power=False) if recent_pa >= MIN_RECENT_HITTER_PA else ""

    if season_text:
        event.stats_lines.append(f"{label} season: {season_text}")
    if recent_text:
        event.stats_lines.append(f"Last 14d: {recent_text}")


# -----------------------------
# Rendering / packing
# -----------------------------
def level_change_text(event: TxnEvent) -> str:
    player = " ".join(x for x in [event.position, event.person_name] if x).strip() or event.person_name
    if event.event_type == "recalled":
        lines = [f"SF recalled {player} from {destination_name(event.from_id, event.from_name)}"]
    elif event.event_type == "optioned":
        lines = [f"SF optioned {player} to {destination_name(event.to_id, event.to_name)}"]
    else:
        lines = [f"{player} → {destination_name(event.to_id, event.to_name)}"]
        if event.from_id:
            lines.append(f"From {destination_name(event.from_id, event.from_name)}")
    lines.extend(event.stats_lines)

    while len("\n".join(lines)) > MAX_CHARS and len(lines) > 1:
        lines.pop()
    return "\n".join(lines)


def _pack_plain_events(events: List[TxnEvent]) -> List[PostBundle]:
    by_header: Dict[str, List[TxnEvent]] = {}
    for event in events:
        by_header.setdefault(event.header, []).append(event)

    sections: List[Tuple[List[str], List[int]]] = []
    for header in SECTION_ORDER:
        items = by_header.get(header, [])
        if not items:
            continue
        items.sort(key=lambda x: (x.sort_date, x.id))
        lines = [header]
        ids: List[int] = []
        for event in items:
            lines.append(f"• {make_compact_line(event)}")
            ids.append(event.id)
        sections.append((lines, ids))

    posts: List[PostBundle] = []
    current_lines: List[str] = []
    current_ids: List[int] = []

    def flush() -> None:
        nonlocal current_lines, current_ids
        if current_lines:
            posts.append(PostBundle("\n".join(current_lines), current_ids[:]))
            current_lines = []
            current_ids = []

    for lines, ids in sections:
        section = "\n".join(lines)
        candidate = ("\n\n".join(["\n".join(current_lines), section]) if current_lines else section)
        if len(candidate) <= MAX_CHARS:
            if current_lines:
                current_lines.append("")
            current_lines.extend(lines)
            current_ids.extend(ids)
            continue

        flush()
        header = lines[0]
        block_lines = [header]
        block_ids: List[int] = []
        for line, event_id in zip(lines[1:], ids):
            candidate = "\n".join(block_lines + [line])
            if len(candidate) > MAX_CHARS and len(block_lines) > 1:
                posts.append(PostBundle("\n".join(block_lines), block_ids[:]))
                block_lines = [f"{header} (cont.)", line]
                block_ids = [event_id]
            else:
                block_lines.append(line)
                block_ids.append(event_id)
        if block_ids:
            current_lines = block_lines
            current_ids = block_ids

    flush()
    return posts


def build_posts(events: List[TxnEvent]) -> List[PostBundle]:
    enriched: List[PostBundle] = []
    plain: List[TxnEvent] = []

    for event in sorted(events, key=lambda x: (x.sort_date, x.id)):
        if event.is_level_change and event.stats_lines:
            enriched.append(PostBundle(level_change_text(event), [event.id]))
        else:
            plain.append(event)

    return enriched + _pack_plain_events(plain)


# -----------------------------
# Bluesky / audit
# -----------------------------
def bsky_login() -> Client:
    client = Client()
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"])
    return client


def append_audit_event(state: Dict[str, Any], event: TxnEvent, post_text: str, post_uri: str = "") -> None:
    state.setdefault("recent_events", []).append(
        {
            "transaction_id": event.id,
            "event_key": event.event_key,
            "effective_date": (event.sort_date or "")[:10],
            "player": event.person_name,
            "type": event.event_type,
            "from_team": destination_name(event.from_id, event.from_name) if event.from_id else "",
            "to_team": destination_name(event.to_id, event.to_name) if event.to_id else "",
            "post_text": post_text,
            "post_uri": post_uri,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def main() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    today = datetime.now(tz).date()
    dry_run = os.getenv("DRY_RUN", "0") == "1"

    override_start = os.getenv("OVERRIDE_START_DATE")
    override_end = os.getenv("OVERRIDE_END_DATE")
    if override_start and override_end:
        start = date.fromisoformat(override_start)
        end = date.fromisoformat(override_end)
    else:
        start = today - timedelta(days=LOOKBACK_DAYS)
        end = today

    state = load_transaction_state()
    seen_ids = {int(x) for x in state.get("seen_transaction_ids", [])}
    seen_event_keys = set(state.get("seen_event_keys", []))

    s = make_session()
    collected: Dict[int, TxnEvent] = {}
    newly_observed_ids: Set[int] = set()
    newly_observed_keys: Set[str] = set()

    for query_team_id in TRACKED_TEAM_IDS:
        for t in fetch_transactions(s, query_team_id, start, end):
            raw_id = t.get("id")
            if raw_id is None:
                continue
            tid = int(raw_id)
            if tid in seen_ids or tid in collected:
                continue

            newly_observed_ids.add(tid)
            desc = normalize(t.get("description", ""))
            dl = desc.lower()
            from_id, from_name, to_id, to_name = get_team_fields(t)

            if is_internal_dsl_move(from_id, to_id, dl):
                continue

            display_team_id = choose_display_team_id(from_id, to_id, query_team_id)
            if display_team_id is None:
                continue
            header = TEAM_HEADER.get(display_team_id)
            if not header:
                continue

            person_obj = t.get("person") or {}
            person_id = person_obj.get("id")
            person_id = int(person_id) if person_id is not None else None
            person_name = person_obj.get("fullName") or ""
            sort_date = pick_sort_date(t)
            event_type = classify_event(desc)
            event_key = make_event_key(sort_date, person_id, event_type, from_id, to_id, desc)

            if event_key in seen_event_keys or event_key in newly_observed_keys:
                continue
            newly_observed_keys.add(event_key)

            event = TxnEvent(
                id=tid,
                sort_date=sort_date,
                person_id=person_id,
                person_name=person_name,
                position=extract_position(desc, person_name),
                event_type=event_type,
                from_id=from_id,
                from_name=from_name,
                to_id=to_id,
                to_name=to_name,
                display_team_id=display_team_id,
                header=header,
                description=desc,
                event_key=event_key,
            )
            attach_stats_context(s, event)
            collected[tid] = event

    if not state.get("bootstrapped", False):
        if dry_run:
            print(f"DRY RUN bootstrap: would mark {len(newly_observed_ids)} transactions seen; no posts.")
            return
        state["bootstrapped"] = True
        state["seen_transaction_ids"] = sorted(seen_ids | newly_observed_ids)
        state["seen_event_keys"] = sorted(seen_event_keys | newly_observed_keys)
        save_transaction_state(state)
        print(f"Bootstrapped: marked {len(newly_observed_ids)} transactions as seen (no posts).")
        return

    posts = build_posts(list(collected.values()))

    if dry_run:
        if not posts:
            print("DRY RUN: no new postable transactions.")
        for i, post in enumerate(posts, start=1):
            print(f"\n--- POST {i}/{len(posts)} ({len(post.text)} chars) ---\n{post.text}")
        print(f"\nDRY RUN: {len(posts)} posts covering {len(collected)} normalized events; state unchanged.")
        return

    if not posts:
        if newly_observed_ids:
            state["seen_transaction_ids"] = sorted(seen_ids | newly_observed_ids)
            state["seen_event_keys"] = sorted(seen_event_keys | newly_observed_keys)
            save_transaction_state(state)
            print(f"Recorded {len(newly_observed_ids)} new transaction IDs; nothing to post.")
        else:
            print("No new transactions.")
        return

    events_by_id = {event.id: event for event in collected.values()}
    collected_ids = set(events_by_id)
    collected_keys = {event.event_key for event in collected.values()}

    # IDs that normalized away (for example duplicate API rows or internal DSL moves)
    # can be persisted immediately. Postable events are marked seen only after their
    # Bluesky post succeeds, so a partial posting failure retries only the unposted work.
    persisted_ids = seen_ids | (newly_observed_ids - collected_ids)
    persisted_keys = seen_event_keys | (newly_observed_keys - collected_keys)
    state["seen_transaction_ids"] = sorted(persisted_ids)
    state["seen_event_keys"] = sorted(persisted_keys)
    if persisted_ids != seen_ids or persisted_keys != seen_event_keys:
        save_transaction_state(state)

    client = bsky_login()

    for post in posts:
        response = client.send_post(text=post.text)
        post_uri = getattr(response, "uri", "") or ""
        for event_id in post.event_ids:
            event = events_by_id.get(event_id)
            if not event:
                continue
            persisted_ids.add(event_id)
            persisted_keys.add(event.event_key)
            append_audit_event(state, event, post.text, post_uri)
        state["seen_transaction_ids"] = sorted(persisted_ids)
        state["seen_event_keys"] = sorted(persisted_keys)
        # Persist after each successful Bluesky post. The workflow commits this even if a later post fails.
        save_transaction_state(state)
        time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    print(f"Posted {len(posts)} posts covering {len(collected)} normalized events.")


if __name__ == "__main__":
    main()
