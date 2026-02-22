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

# team_id -> official team name used in API descriptions
TEAM_NAME: Dict[int, str] = {
    2134: "DSL Giants Black",
    615: "DSL Giants Orange",
    408: "ACL Giants",
    476: "San Jose Giants",
    461: "Eugene Emeralds",
    3410: "Richmond Flying Squirrels",
    105: "Sacramento River Cats",
}

# Display groups (this is what appears as the header)
# Note: DSL combines Orange + Black into one section.
GROUPS: List[Tuple[str, List[int]]] = [
    ("DSL Giants", [615, 2134]),
    ("ACL Giants", [408]),
    ("San Jose", [476]),
    ("Eugene", [461]),
    ("Richmond", [3410]),
    ("Sacramento", [105]),
]

STATE_PATH = "state.json"
API_BASE = "https://statsapi.mlb.com/api/v1"


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
# Transactions
# -----------------------------
@dataclass(frozen=True)
class Txn:
    id: int
    team_id: int
    sort_date: str
    description: str


def fetch_transactions(s: requests.Session, team_id: int, start_date: date, end_date: date) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/transactions"
    params = {"teamId": str(team_id), "startDate": start_date.isoformat(), "endDate": end_date.isoformat()}
    r = s.get(url, params=params, timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("transactions", [])


def normalize(desc: str) -> str:
    return " ".join((desc or "").strip().split())


def pick_sort_date(t: Dict[str, Any]) -> str:
    return (t.get("effectiveDate") or t.get("date") or t.get("resolutionDate") or "")


def _ci_replace(text: str, pattern: str, repl: str) -> str:
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def compact_description(desc: str, own_team_names: List[str]) -> str:
    """
    Reduce redundancy for an 'in-the-know' audience while preserving key info.
    - Remove leading team name if present.
    - Remove "assigned to <own team>" / "to <own team>" / "from <own team>" phrases.
    - Keep mentions of OTHER teams (e.g. "from Eugene", "to Sacramento") intact.
    """
    d = normalize(desc)

    # 1) Remove leading team name (e.g. "Richmond Flying Squirrels ..." -> "...")
    for name in own_team_names:
        if d.lower().startswith(name.lower()):
            d = d[len(name):].lstrip()
            break

    # 2) Remove "assigned to <own team>" / "to <own team>" / "from <own team>"
    # Do this carefully so we don't remove other-team context.
    for name in own_team_names:
        n = re.escape(name)

        # "assigned to (the )?<team>." -> "assigned."
        d = _ci_replace(d, rf"\bassigned to (the )?{n}\b", "assigned")

        # "to (the )?<team>" -> "" (keep rest: e.g. "promoted to X from Y" keeps "from Y")
        d = _ci_replace(d, rf"\bto (the )?{n}\b", "")

        # "from (the )?<team>" -> ""
        d = _ci_replace(d, rf"\bfrom (the )?{n}\b", "")

    # Cleanup extra spaces before punctuation
    d = re.sub(r"\s+\.", ".", d)
    d = re.sub(r"\s+,", ",", d)
    d = re.sub(r"\s{2,}", " ", d).strip()

    return d


# -----------------------------
# Post packing (cram more into fewer posts)
# -----------------------------
def wrap_long_line(line: str, max_len: int) -> List[str]:
    """
    If a single bullet line would exceed max_len, wrap it across multiple lines
    WITHOUT truncating content.
    """
    if len(line) <= max_len:
        return [line]

    words = line.split(" ")
    chunks: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_len:
            cur += " " + w
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks


def build_sections(
    txns_by_team: Dict[int, List[Txn]],
) -> List[List[str]]:
    """
    Returns list of sections, each section is a list of lines:
      [Header, bullet, bullet, ...]
    """
    sections: List[List[str]] = []

    for header, team_ids in GROUPS:
        all_txns: List[Txn] = []
        own_names = [TEAM_NAME[tid] for tid in team_ids if tid in TEAM_NAME]

        for tid in team_ids:
            all_txns.extend(txns_by_team.get(tid, []))

        if not all_txns:
            continue

        all_txns.sort(key=lambda x: (x.sort_date, x.id))

        lines = [header]
        for t in all_txns:
            short = compact_description(t.description, own_names)
            bullet = f"• {short}"
            # Ensure no single line hard-exceeds post constraints by wrapping
            wrapped = wrap_long_line(bullet, MAX_CHARS - 1)  # -1 buffer
            lines.extend(wrapped)

        sections.append(lines)

    return sections


def pack_sections_into_posts(sections: List[List[str]], max_chars: int) -> List[str]:
    """
    Pack multiple sections into as few posts as possible.
    Each section is separated by a blank line.
    """
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
        # +2 for the blank line separator if we already have content
        sep = "\n\n" if cur_lines else ""
        add_len = len(sep) + len(sec_text)

        if cur_len + add_len <= max_chars:
            if sep:
                cur_lines.append("")  # blank line
                cur_len += len(sep)
            cur_lines.extend(sec)
            cur_len += len(sec_text)
        else:
            # Section doesn't fit in current post; flush and try again
            flush()

            # If section itself > max_chars, we must split section across posts.
            # We'll split by lines, preserving full content.
            if len(sec_text) <= max_chars:
                cur_lines = sec[:]
                cur_len = len(sec_text)
            else:
                # line-by-line split
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

    # Optional test override (YYYY-MM-DD)
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

    # Collect new txns by team
    txns_by_team: Dict[int, List[Txn]] = {tid: [] for tid in TEAM_NAME.keys()}
    for team_id in TEAM_NAME.keys():
        for t in fetch_transactions(s, team_id, start, end):
            tid = int(t.get("id"))
            if tid in seen:
                continue
            desc = normalize(t.get("description", ""))
            if not desc:
                continue
            txns_by_team[team_id].append(
                Txn(
                    id=tid,
                    team_id=team_id,
                    sort_date=pick_sort_date(t),
                    description=desc,
                )
            )

        txns_by_team[team_id].sort(key=lambda x: (x.sort_date, x.id))

    new_count = sum(len(v) for v in txns_by_team.values())

    # Bootstrap: mark everything as seen, no posts.
    if not state.get("bootstrapped", False):
        for v in txns_by_team.values():
            for txn in v:
                seen.add(txn.id)
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

    # Build sections (DSL combined) and pack into fewer posts
    sections = build_sections(txns_by_team)
    posts = pack_sections_into_posts(sections, max_chars=MAX_CHARS)

    client = bsky_login()

    # Post everything, marking IDs as seen after success per post batch
    # We’ll mark all discovered IDs once we're done posting (simpler, still safe in practice).
    for text in posts:
        client.send_post(text=text)
        time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    # Mark ALL discovered txns as seen after posting the batch
    for v in txns_by_team.values():
        for txn in v:
            seen.add(txn.id)

    state["seen_transaction_ids"] = sorted(seen)
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"Posted {len(posts)} posts covering {new_count} transactions.")


if __name__ == "__main__":
    main()
