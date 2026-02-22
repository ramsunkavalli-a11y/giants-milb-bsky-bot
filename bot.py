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

MAX_POSTS_PER_RUN = 10
MAX_CHARS = 300
LOOKBACK_DAYS = 14  # rolling two weeks

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


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"bootstrapped": False, "seen_transaction_ids": [], "last_run_iso": None}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass(frozen=True)
class Txn:
    id: int
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
    """
    Prefer effectiveDate for ordering; fall back to date then resolutionDate.
    All are typically YYYY-MM-DD in this API.
    """
    return (t.get("effectiveDate") or t.get("date") or t.get("resolutionDate") or "")


def chunk_lines(lines: List[str], max_chars: int, max_posts: int) -> List[str]:
    posts: List[str] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            posts.append("\n".join(cur))
            cur = []
            cur_len = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        add_len = len(line) + (1 if cur else 0)
        if cur_len + add_len <= max_chars:
            cur.append(line)
            cur_len += add_len
        else:
            flush()
            if len(line) > max_chars:
                posts.append(line[: max_chars - 1] + "…")
            else:
                cur = [line]
                cur_len = len(line)
        if len(posts) >= max_posts:
            break

    if len(posts) < max_posts:
        flush()

    return posts[:max_posts]


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

    new: List[Txn] = []
    for team_id in AFFILIATES:
        txns = fetch_transactions(s, team_id, start, end)
        for t in txns:
            tid = int(t.get("id"))
            if tid in seen:
                continue
            desc = normalize(t.get("description", ""))
            if not desc:
                continue
            new.append(Txn(id=tid, sort_date=pick_sort_date(t), description=desc))

    # Stable ordering: by effective/date then id
    new.sort(key=lambda x: (x.sort_date, x.id))

    # First run bootstrap: mark everything in the window as seen, no posts.
    if not state.get("bootstrapped", False):
        for t in new:
            seen.add(t.id)
        state["bootstrapped"] = True
        state["seen_transaction_ids"] = sorted(seen)
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print(f"Bootstrapped: marked {len(new)} transactions as seen (no posts).")
        return

    if not new:
        state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
        save_state(state)
        print("No new transactions.")
        return

    lines = [t.description for t in new]
    posts = chunk_lines(lines, MAX_CHARS, MAX_POSTS_PER_RUN)

    client = bsky_login()
    for p in posts:
        client.send_post(text=p)
        time.sleep(1.2)

    # Mark all discovered txns as seen to avoid backlog spam
    for t in new:
        seen.add(t.id)

    state["seen_transaction_ids"] = sorted(seen)
    state["last_run_iso"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    print(f"Posted {len(posts)} post(s). Marked {len(new)} txns as seen.")


if __name__ == "__main__":
    main()
