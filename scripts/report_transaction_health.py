"""Write a non-mutating publication-health summary for the Actions run."""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


STATE_PATH = Path("transaction_state.json")
BURST_POST_LIMIT = 10
DELAY_WARNING_DAYS = 1


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    if not STATE_PATH.exists():
        print("::warning::Transaction state file is missing; publication health could not be assessed.")
        return

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = state.get("recent_events", [])
    now = datetime.now(timezone.utc)
    recent = [event for event in events if event.get("recorded_at") and parse_timestamp(event["recorded_at"]) >= now - timedelta(days=1)]
    posts = {event.get("post_uri") or event.get("post_text", "") for event in recent}
    last_post = max((parse_timestamp(event["recorded_at"]) for event in events if event.get("recorded_at")), default=None)

    delayed = []
    for event in events:
        if not event.get("recorded_at") or not event.get("effective_date"):
            continue
        delay = (parse_timestamp(event["recorded_at"]).date() - date.fromisoformat(event["effective_date"])).days
        if delay > DELAY_WARNING_DAYS:
            delayed.append((delay, event))

    summary = [
        "## Transaction publication health",
        f"- State last updated: `{state.get('updated_at', 'unknown')}`",
        f"- Last recorded publication: `{last_post.isoformat() if last_post else 'none'}`",
        f"- Last 24 hours: **{len(posts)} distinct posts** covering **{len(recent)} transaction records**.",
    ]
    if delayed:
        summary.append(f"- Delayed publications (> {DELAY_WARNING_DAYS} day): **{len(delayed)}**.")
        print(f"::warning::{len(delayed)} audited transaction(s) were published more than {DELAY_WARNING_DAYS} day after their effective date.")
    if len(posts) >= BURST_POST_LIMIT:
        summary.append(f"- High-volume run window (threshold: {BURST_POST_LIMIT} posts): review the feed for readability.")
        print(f"::warning::{len(posts)} distinct posts were recorded in the last 24 hours.")

    text = "\n".join(summary) + "\n"
    print(text)
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(text)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"::warning::Could not generate transaction publication-health report: {exc}")
        sys.exit(0)
