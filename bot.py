import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Tuple, Optional, Set

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
ACL = 408
SAN_JOSE = 476
EUGENE = 461
RICHMOND = 3410
SACRAMENTO = 105

TRACKED_TEAM_IDS: Set[int] = {DSL_BLACK, DSL_ORANGE, ACL, SAN_JOSE, EUGENE, RICHMOND, SACRAMENTO}

# Display headers (DSL combines Orange+Black)
TEAM_HEADER: Dict[int, str] = {
    DSL_BLACK: "DSL Giants",
    DSL_ORANGE: "DSL Giants",
    ACL: "ACL Giants",
    SAN_JOSE: "San Jose",
    EUGENE: "Eugene",
    RICHMOND: "Richmond",
    SACRAMENTO: "Sacramento",
}

# Short names used in "from X"
TEAM_SHORT: Dict[int, str] = {
    DSL_BLACK: "DSL",
    DSL_ORANGE: "DSL",
    ACL: "ACL",
    SAN_JOSE: "San Jose",
    EUGENE: "Eugene",
    RICHMOND: "Richmond",
    SACRAMENTO: "Sacramento",
    137: "SF",
}

# Order of sections inside posts
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
    header: str
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
    n = re.sub(r"\bFlying Squirrels\b", "", n).strip()
    n = re.sub(r"\bRiver Cats\b", "", n).strip()
    n = re.sub(r"\bEmeralds\b", "", n).strip()
    n = re.sub(r"\bGiants\b", "", n).strip()
    return n or "?"


def is_internal_dsl_move(from_id: Optional[int], to_id: Optional[int], desc_lower: str) -> bool:
    # Prefer structured
    if (from_id in {DSL_BLACK, DSL_ORANGE}) and (to_id in {DSL_BLACK, DSL_ORANGE}) and (from_id != to_id):
        return True
    # Fallback: description contains both teams
    if ("dsl giants black" in desc_lower) and ("dsl giants orange" in desc_lower):
        return True
    return False


def choose_display_team_id(from_id: Optional[int], to_id: Optional[int], query_team_id: int) -> Optional[int]:
    """
    Prevent duplicates:
    - If toTeam is a tracked affiliate => show under toTeam only.
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


def parse_from_team_from_description(desc: str) -> Optional[str]:
    d = normalize(desc)
    m = re.search(r"\bfrom\s+([A-Za-z0-9 .'-]+)\.?\s*$", d)
    if not m:
        return None
    s = m.group(1).strip()
    s = re.sub(r"\bFlying Squirrels\b", "", s).strip()
    s = re.sub(r"\bRiver Cats\b", "", s).strip()
    s = re.sub(r"\bEmeralds\b", "", s).strip()
    s = re.sub(r"\bGiants\b", "", s).strip()
    return s or None


def replace_injured_list_with_il(desc: str) -> str:
    # Replace "injured list" with "IL" in a few common phrasings
    d = desc
    d = re.sub(r"\binjured list\b", "IL", d, flags=re.IGNORECASE)
    d = re.sub(r"\b7-day IL\b", "7-day IL", d, flags=re.IGNORECASE)
    d = re.sub(r"\b60-day IL\b", "60-day IL", d, flags=re.IGNORECASE)
    d = re.sub(r"\b10-day IL\b", "10-day IL", d, flags=re.IGNORECASE)
    d = re.sub(r"\b15-day IL\b", "15-day IL", d, flags=re.IGNORECASE)
    return normalize(d)


def name_with_pos(desc: str, full_name: str) -> str:
    """
    Extract a position token directly before player's name in description:
      '... RHP Trey Dillard ...' => 'RHP Trey Dillard'
      '... 2B Israel Alfonse ...' => '2B Israel Alfonse'
    """
    if not desc or not full_name:
        return full_name

    d = normalize(desc)
    name_pat = re.escape(full_name)

    m = re.search(rf"\b([A-Z]{{1,3}}HP|[1-3]B|SS|LF|CF|RF|OF|IF|C|P|DH|INF)\s+{name_pat}\b", d)
    if m:
        return f"{m.group(1)} {full_name}"

    return full_name


def make_compact_line(
    desc: str,
    person_name: str,
    header: str,
    from_id: Optional[int],
    from_name: Optional[str],
    to_id: Optional[int],
    to_name: Optional[str],
) -> str:
    d = replace_injured_list_with_il(normalize(desc))
    dl = d.lower()

    is_assigned = (" assigned " in f" {dl} ") or dl.startswith("assigned ")

    dest_names: List[str] = []
    if to_name:
        dest_names.append(to_name)
    if header == "DSL Giants":
        dest_names.extend(["DSL Giants Orange", "DSL Giants Black"])

    origin: Optional[str] = None
    if from_id and (to_id is None or from_id != to_id):
        origin = short_team(from_id, from_name)
    elif from_name and (to_name is None or from_name.lower() != to_name.lower()):
        origin = short_team(from_id, from_name)

    parsed_from = parse_from_team_from_description(d)

    if is_assigned:
        # Strip "assigned to <dest>" redundancy
        for dn in dest_names:
            if dn:
                d = re.sub(rf"\bassigned to (the )?{re.escape(dn)}\b", "assigned", d, flags=re.IGNORECASE)

        origin_final = origin or parsed_from
        if origin_final:
            return f"{person_name} assigned from {origin_final}."
        return f"{person_name} assigned."

    # Non-assignment: strip leading redundancy
    if d.lower().startswith(header.lower()):
        d = d[len(header):].lstrip()

    if header == "DSL Giants":
        for prefix in ("DSL Giants Black", "DSL Giants Orange"):
            if d.lower().startswith(prefix.lower()):
                d = d[len(prefix):].lstrip()
                break

    d = re.sub(r"\s+\.", ".", d).strip()
    return normalize(d)


# -----------------------------
# Packing into fewer posts
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


def build_sections(lines: List[TxnLine]) -> List[List[str]]:
    by_header: Dict[str, List[TxnLine]] = {}
    for tl in lines:
        by_header.setdefault(tl.header, []).append(tl)

    sections: List[List[str]] = []
    for header in SECTION_ORDER:
        items = by_header.get(header, [])
        if not items:
            continue
        items.sort(key=lambda x: (x.sort_date, x.id))
        sec = [header]
        for tl in items:
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

    collected: Dict[int, TxnLine] = {}
    discovered_ids: List[int] = []

    for query_team_id in TRACKED_TEAM_IDS:
        txns = fetch_transactions(s, query_team_id, start, end)
        for t in txns:
            tid = int(t.get("id"))
            if tid in seen or tid in collected:
                continue

            desc = normalize(t.get("description", ""))
            dl = desc.lower()

            from_id, from_name, to_id, to_name = get_team_fields(t)

            # Skip internal DSL moves entirely
            if is_internal_dsl_move(from_id, to_id, dl):
                seen.add(tid)
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
            sort_date = pick_sort_date(t)

            # Include position if possible
            person_for_line = name_with_pos(desc, person) if person else desc

            text = make_compact_line(
                desc=desc,
                person_name=person_for_line,
                header=header,
                from_id=from_id,
                from_name=from_name,
                to_id=to_id,
                to_name=to_name,
            )

            collected[tid] = TxnLine(
                id=tid,
                sort_date=sort_date,
                display_team_id=display_team_id,
                header=header,
                text=text,
            )
            discovered_ids.append(tid)

    new_count = len(collected)

    if not state.get("bootstrapped", False):
        for tid in collected.keys():
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

    sections = build_sections(list(collected.values()))
    posts = pack_sections_into_posts(sections, max_chars=MAX_CHARS)

    client = bsky_login()
    for text in posts:
        client.send_post(text=text)
        time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    for tid in collected.keys():
        seen.add(tid)

    state["seen_transaction_ids"] = sorted(seen)
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"Posted {len(posts)} posts covering {new_count} transactions.")


if __name__ == "__main__":
    main()
