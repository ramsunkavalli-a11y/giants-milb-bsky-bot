# giants-milb-bsky-bot

Bluesky automation for San Francisco Giants minor-league transactions.

## Production bot

`bot.py` polls MLB's transaction feed for Giants affiliates and posts new organization moves to Bluesky.

The bot treats MLB's transaction as the fact and adds context without guessing organizational intent:

- affiliate assignments are described neutrally rather than automatically labeled promotions/demotions;
- level changes can include the player's season line at the departing team plus a meaningful last-14-days split;
- hitters use AVG/OBP/SLG + PA (and HR on the season line); pitchers use IP/ERA + K%/BB%;
- MLB stat queries are scoped to the departing affiliate and MiLB level;
- duplicate-looking MLB rows are deduped by both transaction ID and normalized event key;
- IL/releases/other routine moves remain compact and can be grouped by affiliate;
- `transaction_state.json` is the only persistent bot state;
- empty hourly runs do not create state-only commits;
- state is persisted after each successful Bluesky post so partial failures retry only unposted work.

See [`docs/TRANSACTIONS.md`](docs/TRANSACTIONS.md) for formatting rules, stat thresholds, state behavior, and validation details.

## Local / manual validation

```bash
pip install -r requirements-transactions.txt
python -m unittest -v test_bot.py
DRY_RUN=1 python bot.py
python scripts/validate_transaction_stats.py
```

Manual GitHub Actions runs of **Giants MiLB Transactions Bot** default to dry-run mode. Scheduled runs remain live.

## Repository scope

This repository supports the MiLB transactions bot only. The previous DSL Orange box-score/recap implementation was removed rather than retained as legacy code; any future game/recap bot should be rebuilt as a separate, intentionally designed implementation.

## Bluesky profile

Set the account display name to **Giants MiLB Transactions** and use this bio:

> Automated San Francisco Giants minor-league roster moves. MLB StatsAPI; factual updates, no rumor or editorial judgment.
