# giants-milb-bsky-bot

Bluesky automation for San Francisco Giants minor-league updates.

## Active production bot: MiLB transactions

`bot.py` polls MLB's transaction feed for Giants affiliates and posts new organization moves to Bluesky.

The bot now treats MLB's transaction as the fact and adds context without guessing organizational intent:

- affiliate assignments are described neutrally rather than automatically labeled promotions/demotions;
- level changes can include the player's season line at the departing team plus a meaningful last-14-days split;
- hitters use AVG/OBP/SLG + PA (and HR on the season line); pitchers use IP/ERA + K%/BB%;
- duplicate-looking MLB rows are deduped by both transaction ID and normalized event key;
- IL/releases/other routine moves remain compact and can be grouped by affiliate;
- transaction state and recent audit history live in `transaction_state.json` once production creates it;
- empty hourly runs do not create state-only commits.

See [`docs/TRANSACTIONS.md`](docs/TRANSACTIONS.md) for the formatting rules, stats thresholds, state migration, and validation details.

### Local / manual checks

```bash
pip install -r requirements-transactions.txt
python -m unittest -v test_bot.py
DRY_RUN=1 python bot.py
python scripts/validate_transaction_stats.py
```

Manual GitHub Actions runs of **Giants MiLB Transactions Bot** default to dry-run mode. Scheduled runs remain live.

## Legacy DSL Orange tools

The repository also contains the older DSL Giants Orange box-score and daily-recap code (`gameday_dsl_orange.py`) and its workflows. Those tools are separate from the active transaction workflow.

### DSL local examples

```bash
python gameday_dsl_orange.py
DRY_RUN=1 OVERRIDE_GAMEPK=811804 python gameday_dsl_orange.py
DRY_RUN=1 OVERRIDE_DATE=2025-07-18 python gameday_dsl_orange.py
python gameday_dsl_orange.py --recap
```

### DSL supporting files

- `state.json` remains the legacy DSL state file and the migration source for historical transaction IDs.
- `prospects.json` is used by DSL post highlighting.
- `player_cache.json` caches DSL player metadata.
- `data/tango_we.json` is the vendored win-expectancy lookup table.
- `templates/boxscore_card.html` controls the DSL box-score card.
