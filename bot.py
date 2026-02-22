import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Any, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from atproto import Client

# -----------------------------
# Config
# -----------------------------
MAX_CHARS = 300
LOOKBACK_DAYS = 14  # rolling two weeks
SLEEP_BETWEEN_POSTS_SEC = 1.2

# Ordered from DSL -> AAA (as you wanted, DSL to Sacramento)
AFFILIATES: Dict[int, str] = {
    2134: "DSL Giants Black",
    615: "DSL Giants Orange",
    408: "ACL Giants",
    476: "San Jose (A)",
    461: "Eugene (High-A)",
    3410: "Richmond (AA)",
    105: "Sacramento (AAA)",
}

STATE_PATH = "state.json"
API_BASE = "https://statsapi.mlb.com/api/v1"


# -----------------------------
# HTTP helpers (retries)
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
    s.headers.update({"User-Agent": "giants-affiliates-bot/1.0"})
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
# Transactions fetch
# -----------------------------
@dataclass(frozen=True)
class Txn:
    id: int
    team_id: int
    team_label: str
    sort_date: str  # YYYY-MM-DD (best effort)
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
    # Prefer effectiveDate for ordering; fall back to date then resolutionDate
    return (t.get("effectiveDate") or t.get("date") or t.get("resolutionDate") or "")


def strip_team_prefix(team_label: str, desc: str) -> str:
    """
    If description starts with the affiliate name, remove it to avoid redundancy.
    Example:
      "DSL Giants Black activated C X ..." -> "activated C X ..."
    """
    d = desc.strip()
    tl = team_label.strip()
    if not tl:
        return d
    if d.lower().startswith(tl.lower()):
        d = d[len(tl):].lstrip()
        # common pattern: "<team> activated..." so now it starts with verb
    return d


def collect_new_by_affiliate(
    s: requests.Session,
    seen_ids: set,
    start_date: date,
    end_date: date,
) -> Dict[int, List[Txn]]:
    out: Dict[int, List[Txn]] = {tid: [] for tid in AFFILIATES.keys()}

    for team_id, label in AFFILIATES.items():
        txns = fetch_transactions(s, team_id, start_date, end_date)
        for t in txns:
            tid = int(t.get("id"))
            if tid in seen_ids:
                continue
            desc = normalize(t.get("description", ""))
            if not desc:
                continue

            out[team_id].append(
                Txn(
                    id=tid,
                    team_id=team_id,
                    team_label=label,
                    sort_date=pick_sort_date(t),
                    description=desc,
                )
            )

        # Stable ordering inside each affiliate
        out[team_id].sort(key=lambda x: (x.sort_date, x.id))

    return out


# -----------------------------
# Post building / chunking
# -----------------------------
def build_affiliate_lines(team_label: str, txns: List[Txn]) -> List[Tuple[str, int]]:
    """
    Returns list of (line_text, txn_id) for this affiliate.
    """
    lines: List[Tuple[str, int]] = []
    for t in txns:
        short = strip_team_prefix(team_label, t.description)
        # bullet for readability
        line = f"• {short}"
        lines.append((line, t.id))
    return lines


def chunk_affiliate_posts(
    header: str,
    lines_with_ids: List[Tuple[str, int]],
    max_chars: int,
) -> List[Tuple[str, List[int]]]:
    """
    Splits one affiliate's lines into multiple posts.
    Each post returns (text, txn_ids_included).
    Never drops lines.
    """
    posts: List[Tuple[str, List[int]]] = []

    def make_header(is_cont: bool) -> str:
        return f"{header} (cont.)" if is_cont else header

    i = 0
    cont = False
    while i < len(lines_with_ids):
        hdr = make_header(cont)
        # Start post with header
        text_lines = [hdr]
        ids_in_post: List[int] = []

        cur_len = len(hdr)
        # Add lines until full
        while i < len(lines_with_ids):
            line, tid = lines_with_ids[i]
            add = "\n" + line
            if cur_len + len(add) <= max_chars:
                text_lines.append(line)
                ids_in_post.append(tid)
                cur_len += len(add)
                i += 1
            else:
                # If a single line can't fit even in an empty post (rare), truncate that line safely
                # but DO NOT drop the transaction.
                if cur_len == len(hdr):
                    # allow header + truncated line
                    budget = max_chars - (len(hdr) + 1)  # minus newline
                    if budget <= 1:
                        # Extreme edge case: header alone fills the post. Make a minimal post then continue.
                        posts.append(("\n".join(text_lines)[:max_chars], ids_in_post))
                        cont = True
                        break
                    truncated = (line[: budget - 1] + "…") if len(line) > budget else line
                    text_lines.append(truncated)
                    ids_in_post.append(tid)
                    i += 1
                break

        posts.append(("\n".join(text_lines), ids_in_post))
        cont = True

    return posts


def build_all_posts_grouped(
    txns_by_team: Dict[int, List[Txn]],
    max_chars: int,
) -> List[Tuple[str, List[int]]]:
    """
    Returns list of (post_text, txn_ids_included), grouped by affiliate
    in the order of AFFILIATES.
    """
    all_posts: List[Tuple[str, List[int]]] = []

    for team_id, label in AFFILIATES.items():
        txns = txns_by_team.get(team_id, [])
        if not txns:
            continue

        lines_with_ids = build_affiliate_lines(label, txns)
        posts = chunk_affiliate_posts(label, lines_with_ids, max_chars=max_chars)
        all_posts.extend(posts)

    return all_posts


# -----------------------------
# Bluesky posting
# -----------------------------
def bsky_login() -> Client:
    client = Client()
    client.login(os.environ["BSKY_HANDLE"], os.environ["BSKY_APP_PASSWORD"])
    return client


def main() -> None:
    tz = ZoneInfo("America/Los_Angeles")
    today = datetime.now(tz).date()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today

    state = load_state()
    seen = set(state.get("seen_transaction_ids", []))

    s = make_session()

    txns_by_team = collect_new_by_affiliate(s, seen, start, end)

    # Flatten count
    new_count = sum(len(v) for v in txns_by_team.values())

    # First run bootstrap: mark everything in the window as seen, no posts.
    if not state.get("bootstrapped", False):
        for v in txns_by_team.values():
            for t in v:
                seen.add(t.id)
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

    # Build ALL posts needed (no cap)
    posts_with_ids = build_all_posts_grouped(txns_by_team, max_chars=MAX_CHARS)

    client = bsky_login()

    # Post everything, and only mark IDs seen after the post that contains them succeeds
    posted_posts = 0
    posted_txn_ids: List[int] = []

    for text, ids_in_post in posts_with_ids:
        client.send_post(text=text)
        posted_posts += 1
        posted_txn_ids.extend(ids_in_post)

        # Mark these as seen now (so a failure later doesn't cause partial repost spam)
        for tid in ids_in_post:
            seen.add(tid)

        time.sleep(SLEEP_BETWEEN_POSTS_SEC)

    state["seen_transaction_ids"] = sorted(seen)
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"Posted {posted_posts} posts, covering {len(set(posted_txn_ids))} transactions.")


if __name__ == "__main__":
    main()
