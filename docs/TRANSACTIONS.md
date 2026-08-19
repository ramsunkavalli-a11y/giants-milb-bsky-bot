# MiLB transactions bot

This document describes the active Giants MiLB transactions bot (`bot.py`). The DSL game/recap code is separate and is not part of this workflow.

## Purpose

Turn MLB's structured transaction log into concise Giants-organization updates with enough context to make level changes useful without inventing organizational intent.

Core rule: **the bot reports the transaction and adds factual context; it does not infer why the Giants made the move.** Moving to a higher level is not automatically called a promotion, and moving to a lower level is not automatically called a demotion.

## Source and scope

- MLB StatsAPI transactions endpoint.
- Giants affiliates: DSL Black/Orange, ACL, San Jose, Eugene, Richmond, Sacramento.
- San Francisco is recognized when a transaction crosses the MLB/Triple-A boundary.
- Normal discovery lookback is 14 days; MLB transaction IDs plus a normalized event key prevent reposts.
- Internal DSL Black/Orange reshuffling remains suppressed.

## Level-change context

For an assignment, option, or recall between organization levels, the bot attempts to fetch stats from the team the player is leaving, through the day before the transaction.

Hitters:
- season line at departing team: AVG/OBP/SLG, PA, HR when nonzero;
- last 14 days: AVG/OBP/SLG and PA;
- last-14 line is omitted below 20 PA.

Pitchers:
- season line at departing team: IP, ERA, K%, BB%;
- last 14 days: IP, ERA, K%, BB%;
- last-14 line is omitted below 5 IP.

Stats are context only. A hot or cold 14-day line is never presented as the reason for a transaction. If the MLB stats request fails, the transaction still posts in the compact fallback format.

### Example: assignment with hitter context

```text
SS Jhonny Level → High-A Eugene
From Low-A San Jose
SJ season: .325/.392/.576 in 280 PA, 10 HR
Last 14d: .347/.418/.612 in 55 PA
```

### Example: assignment with pitcher context

```text
RHP Example Player → Triple-A Sacramento
From Double-A Richmond
RIC season: 72.1 IP, 3.48 ERA · 27.0% K, 8.0% BB
Last 14d: 12.0 IP, 2.25 ERA · 31.0% K, 6.0% BB
```

The numbers above illustrate formatting; production posts use live MLB data.

## Other transactions

IL moves, activations, releases, signings, and other administrative transactions stay compact and can share a post by affiliate. Full affiliate names such as `Richmond Flying Squirrels` are stripped when the affiliate is already the section header.

Explicit MLB transaction language is preserved where it matters. For example, a true option or recall may say `optioned` or `recalled`; generic affiliate assignments remain neutral.

## Dedupe and state

`transaction_state.json` is the active bot's persistent state. On first use it migrates the legacy `seen_transaction_ids` from `state.json`, so existing transactions are not reposted.

State stores:
- MLB transaction IDs;
- normalized event keys, which catch equivalent duplicate transaction rows with different IDs;
- a capped recent-event audit trail with the normalized transaction, rendered post text, and Bluesky URI when available.

The bot no longer writes a `last_run` timestamp on every empty hourly run. A no-news run therefore creates no repository commit.

State is saved after each successful Bluesky post. The workflow's state-commit step runs even if a later post fails, reducing the chance that an already-published post is repeated on the next run.

## Workflows and testing

The active hourly workflow is `.github/workflows/bot.yml` and installs only `requirements-transactions.txt`; the heavier DSL rendering dependencies are not installed hourly.

Manual runs default to dry-run mode. Dry run performs discovery and formatting but does not log into Bluesky or mutate state.

Validation:
- `python -m unittest -v test_bot.py` covers parsing, dedupe, formatting, stats thresholds, and packing.
- `scripts/validate_transaction_stats.py` is a live MLB StatsAPI smoke test using historical Giants-org hitter and pitcher stints.
- `.github/workflows/transactions-validation.yml` runs both checks plus a live bot dry run on transaction-bot pull requests.
