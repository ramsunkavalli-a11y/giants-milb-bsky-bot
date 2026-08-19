import sys
from datetime import date
from pathlib import Path

# Allow direct execution as `python scripts/validate_transaction_stats.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import EUGENE, fetch_date_range_stats, format_hitter_stats, format_pitcher_stats, make_session


def main() -> None:
    session = make_session()

    # Historical Giants-org examples with stable IDs and known 2026 Eugene stints.
    # These calls use the exact production helper, including sportId + teamId.
    kilen = fetch_date_range_stats(
        session,
        person_id=702264,
        group="hitting",
        team_id=EUGENE,
        start_date=date(2026, 4, 3),
        end_date=date(2026, 7, 16),
    )
    switalski = fetch_date_range_stats(
        session,
        person_id=802000,
        group="pitching",
        team_id=EUGENE,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 7, 6),
    )

    assert int(float(kilen.get("plateAppearances") or 0)) > 50, kilen
    assert kilen.get("avg") and kilen.get("obp") and kilen.get("slg"), kilen
    assert float(switalski.get("inningsPitched") or 0) > 50, switalski
    assert switalski.get("era") and switalski.get("battersFaced"), switalski

    print("Kilen Eugene:", format_hitter_stats(kilen))
    print("Switalski Eugene:", format_pitcher_stats(switalski))


if __name__ == "__main__":
    main()
