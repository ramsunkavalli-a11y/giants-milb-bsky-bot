import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Tuple, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from atproto import Client

# -----------------------------
# Config
# -----------------------------
MAX_CHARS = 300
LOOKBACK_DAYS = 14
SLEEP_BETWEEN_POSTS_SEC = 1.2

STATE_PATH = "state.json"
API_BASE = "https://statsapi.mlb.com/api/v1"

# Giants affiliates we care about
DSL_BLACK = 2134
DSL_ORANGE = 615
TRACKED_TEAM_IDS = {DSL_BLACK, DSL_ORANGE, 408, 476, 461, 3410, 105}

# Headers (display)
TEAM_HEADER: Dict[int, str] = {
    DSL_BLACK: "DSL Giants",    # combined
    DSL_ORANGE: "DSL Giants",   # combined
    408: "ACL Giants",
    476: "San Jose",
    461: "Eugene",
    3410: "Richmond",
    105: "Sacramento",
}

# Short names used in "from X"
TEAM_SHORT: Dict[int, str] = {
    DSL_BLACK: "DSL",
    DSL_ORANGE: "DSL",
    408: "ACL",
    476: "San Jose",
    461: "Eugene",
    3410: "Richmond",
    105: "Sacramento",
    137: "SF",
}

SECTION_ORDER = ["DSL Giants", "ACL Giants", "San Jose", "Eugene", "Richmond", "Sacramento"]


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
    s.headers.update({"User-Agent": "giants-milb-transactions-bot/1.0"})
    return s


# -----------------------------
# State
# -----------------------------
def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"bootstrapped": False, "seen_transaction_ids": [], "last_run_iso": None}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class TxnLine:
    id: int
    sort_date: str
    display_team_id: int
    text: str


# -----------------------------
# Fetch / normalize
# -----------------------------
def fetch_transactions(s: requests.Session, team_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/transactions"
    params = {"teamId": str(team_id), "startDate": start_date.isoformat(), "endDate": end_date.isoformat()}
    r = s.get(url, params=params, timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("transactions", [])


def normalize(text: str) -> str:
    return " ".join((text or "").strip().split())


def pick_sort_date(t: Dict[str, Any]) -> str:
    return (t.get("effectiveDate") or t.get("date") or t.get("resolutionDate") or "")


def get_team_fields(t: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    f = t.get("fromTeam") or {}
    to = t.get("toTeam") or {}
    return f.get("id"), f.get("name"), to.get("id"), to.get("name")


def short_team(team_id: Optional[int], team_name: Optional[str]) -> str:
    if team_id is not None and int(team_id) in TEAM_SHORT:
        return TEAM_SHORT[int(team_id)]
    n = normalize(team_name or "")
    # fallback: strip common suffixes
    n = re.sub(r"\bFlying Squirrels\b", "", n).strip()
    n = re.sub(r"\bRiver Cats\b", "", n).strip()
    n = re.sub(r"\bEmeralds\b", "", n).strip()
    n = re.sub(r"\bGiants\b", "", n).strip()
    return n or "?"


def is_internal_dsl_move(from_id: Optional[int], to_id: Optional[int]) -> bool:
    return (from_id in {DSL_BLACK, DSL_ORANGE}) and (to_id in {DSL_BLACK, DSL_ORANGE}) and (from_id != to_id)


def choose_display_team_id(from_id: Optional[int], to_id: Optional[int], query_team_id: int) -> Optional[int]:
    """
    Prevent duplicates:
    - If toTeam is tracked => show under toTeam only.
    - Else if fromTeam is tracked => show under fromTeam.
    - Else fallback to queried team if tracked.
    """
    if to_id in TRACKED_TEAM_IDS:
        return int(to_id)
    if from_id in TRACKED_TEAM_IDS:
        return int(from_id)
    if query_team_id in TRACKED_TEAM_IDS:
        return int(query_team_id)
    return None


def format_assignment(person: str, from_id: Optional[int], from_name: Optional[str]) -> str:
    if from_id or from_name:
        return f"{person} assigned from {short_team(from_id, from_name)}."
    return f"{person} assigned."


def make_compact_line(
    desc: str,
    person_name: str,
    header: str,
    from_id: Optional[int],
    from_name: Optional[str],
    to_id: Optional[int],
) -> str:
    """
    Priority formatting:
    - If looks like 'assigned' and there is a real fromTeam != toTeam => 'Name assigned from X.'
    - Else: strip leading team name (header), and for DSL combined remove Orange/Black prefix if present.
    """
    d = normalize(desc)

    # Assignment detection based on description + presence of from/to teams
    dl = d.lower()
    if (" assigned " in f" {dl} ") and from_id and to_id and from_id != to_id:
        return format_assignment(person_name, from_id, from_name)

    # Otherwise: reduce redundancy but keep meaning
    if d.lower().startswith(header.lower()):
        d = d[len(header):].lstrip()

    if header == "DSL Giants":
        # remove leading "DSL Giants Black/Orange" if it appears
        for prefix in ("DSL Giants Black", "DSL Giants Orange"):
            if d.lower().startswith(prefix.lower()):
                d = d[len(prefix):].lstrip()
                break

    d = re.sub(r"\s+\.", ".", d).strip()
    return d


# -----------------------------
# Packing into fewer posts (cram)
# -----------------------------
def wrap_long_line(line: str, max_len: int) -> List[str]:
    if len(line) <= max_len:
        return [line]
    words = line.split(" ")
    chunks: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_len:
            cur += " " + w
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks


def build_sections(lines_by_header: Dict[str, List[TxnLine]]) -> List[List[str]]:
    sections: List[List[str]] = []
    for header in SECTION_ORDER:
        lines = lines_by_header.get(header, [])
        if not lines:
            continue
        lines.sort(key=lambda x: (x.sort_date, x.id))
        sec = [header]
        for tl in lines:
            bullet = f"• {tl.text}"
            sec.extend(wrap_long_line(bullet, MAX_CHARS - 1))
        sections.append(sec)
    return sections


def pack_sections_into_posts(sections: List[List[str]], max_chars: int) -> List[str]:
    posts: List[str] = []
    cur_lines: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur_lines, cur_len
        if cur_lines:
            posts.append("\n".join(cur_lines))
            cur_lines = []
            cur_len = 0

    for sec in sections:
        sec_text = "\n".join(sec)
        sep = "\n\n" if cur_lines else ""
        add_len = len(sep) + len(sec_text)

        if cur_len + add_len <= max_chars:
            if sep:
                cur_lines.append("")
                cur_len += len(sep)
            cur_lines.extend(sec)
            cur_len += len(sec_text)
        else:
            flush()
            if len(sec_text) <= max_chars:
                cur_lines = sec[:]
                cur_len = len(sec_text)
            else:
                # split big section line-by-line across posts
                tmp: List[str] = []
                tmp_len = 0
                for line in sec:
                    add = (("\n" if tmp else "") + line)
                    if tmp_len + len(add) <= max_chars:
                        tmp.append(line)
                        tmp_len += len(add)
                    else:
                        posts.append("\n".join(tmp))
                        tmp = [line]
                        tmp_len = len(line)
                if tmp:
                    cur_lines = tmp
                    cur_len = tmp_len

    flush()
    return posts


# -----------------------------
# Bluesky
# -----------------------------
def bsky_login() -> Client:
    client = Client()
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"])
    return client


def main() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    today = datetime.now(tz).date()

    override_start = os.getenv("OVERRIDE_START_DATE")
    override_end = os.getenv("OVERRIDE_END_DATE")

    if override_start and override_end:
        start = date.fromisoformat(override_start)
        end = date.fromisoformat(override_end)
    else:
        start = today - timedelta(days=LOOKBACK_DAYS)
        end = today

    state = load_state()
    seen = set(state.get("seen_transaction_ids", []))

    s = make_session()

    # Collect + bucket each txn to ONE display team (destination if available)
    lines_by_header: Dict[str, List[TxnLine]] = {}
    discovered_ids: List[int] = []

    # Query each tracked team (the API expects a teamId filter)
    for query_team_id in TRACKED_TEAM_IDS:
        txns = fetch_transactions(s, query_team_id, start, end)
        for t in txns:
            tid = int(t.get("id"))
            if tid in seen:
                continue

            from_id, from_name, to_id, to_name = get_team_fields(t)

            # Skip internal DSL Orange<->Black moves entirely
            if is_internal_dsl_move(from_id, to_id):
                seen.add(tid)  # mark as seen so we don't keep reconsidering it
                continue

            display_team_id = choose_display_team_id(from_id, to_id, query_team_id)
            if display_team_id is None:
                seen.add(tid)
                continue

            header = TEAM_HEADER.get(display_team_id)
            if not header:
                seen.add(tid)
                continue

            person = (t.get("person") or {}).get("fullName") or ""
            desc = normalize(t.get("description", ""))
            if not desc or not person:
                # If no person name, fallback to raw description
                person = person or ""
                desc = desc or ""
            sort_date = pick_sort_date(t)

            text = make_compact_line(
                desc=desc,
                person_name=person if person else desc,
                header=header,
                from_id=from_id,
                from_name=from_name,
                to_id=to_id,
            )
            text = normalize(text)

            # Store line
            lines_by_header.setdefault(header, []).append(
                TxnLine(
                    id=tid,
                    sort_date=sort_date,
                    display_team_id=display_team_id,
                    text=text,
                )
            )

            discovered_ids.append(tid)

    new_count = len(set(discovered_ids))

    # Bootstrap: mark everything seen, no posts.
    if not state.get("bootstrapped", False):
        for tid in discovered_ids:
            seen.add(tid)
        state["bootstrapped"] = True
        state["seen_transaction_ids"] = sorted(seen)
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"Bootstrapped: marked {new_count} transactions as seen (no posts).")
        return

    if new_count == 0:
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print("No new transactions.")
        return

    sections = build_sections(lines_by_header)
    posts = pack_sections_into_posts(sections, max_chars=MAX_CHARS)

    client = bsky_login()
    for text in posts:
        client.send_post(text=text)
        time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    # Mark as seen after posting
    for tid in discovered_ids:
        seen.add(tid)

    state["seen_transaction_ids"] = sorted(seen)
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"Posted {len(posts)} posts covering {new_count} transactions.")


if __name__ == "__main__":
    main()
