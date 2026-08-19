import sys
from datetime import date
from pathlib import Path

# Allow direct execution as `python scripts/validate_transaction_stats.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import API_BASE, EUGENE, make_session


def stats_payload(session, person_id: int, **params):
    response = session.get(f"{API_BASE}/people/{person_id}/stats", params=params, timeout=30)
    response.raise_for_status()
    return response.json() or {}


def first_stat(payload):
    stats = payload.get("stats") or []
    if not stats:
        return {}
    splits = stats[0].get("splits") or []
    return (splits[0].get("stat") or {}) if splits else {}


def main() -> None:
    session = make_session()

    # MiLB player stats require the sport/level context. Verify both aggregate
    # by-date-range and game-log shapes against historical Giants-org examples.
    kilen_range_payload = stats_payload(
        session,
        702264,
        stats="byDateRange",
        group="hitting",
        sportId="13",
        startDate=date(2026, 4, 3).isoformat(),
        endDate=date(2026, 7, 16).isoformat(),
    )
    kilen_range = first_stat(kilen_range_payload)

    kilen_log_payload = stats_payload(
        session,
        702264,
        stats="gameLog",
        group="hitting",
        sportId="13",
        season="2026",
    )
    kilen_splits = ((kilen_log_payload.get("stats") or [{}])[0].get("splits") or [])
    kilen_team_ids = sorted({(split.get("team") or {}).get("id") for split in kilen_splits if (split.get("team") or {}).get("id")})

    switalski_log_payload = stats_payload(
        session,
        802000,
        stats="gameLog",
        group="pitching",
        sportId="13",
        season="2026",
    )
    switalski_splits = ((switalski_log_payload.get("stats") or [{}])[0].get("splits") or [])
    switalski_team_ids = sorted({(split.get("team") or {}).get("id") for split in switalski_splits if (split.get("team") or {}).get("id")})

    print("Kilen byDateRange:", kilen_range)
    print("Kilen gameLog splits/team IDs:", len(kilen_splits), kilen_team_ids)
    print("Switalski gameLog splits/team IDs:", len(switalski_splits), switalski_team_ids)

    assert int(float(kilen_range.get("plateAppearances") or 0)) > 50, kilen_range_payload
    assert len(kilen_splits) > 20, kilen_log_payload
    assert EUGENE in kilen_team_ids, kilen_team_ids
    assert len(switalski_splits) > 10, switalski_log_payload
    assert EUGENE in switalski_team_ids, switalski_team_ids


if __name__ == "__main__":
    main()
