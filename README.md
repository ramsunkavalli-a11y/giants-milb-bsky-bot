# giants-milb-bsky-bot

Bluesky automation for Giants MiLB updates:
- Transactions bot (`bot.py`)
- DSL Giants Orange automated final/suspended box score bot (`gameday_dsl_orange.py`)
- DSL Giants Orange daily recap mode (`gameday_dsl_orange.py --recap`)

## Local run

### Transactions
```bash
python bot.py
```

### DSL Orange gameday finals/suspended
```bash
python gameday_dsl_orange.py
```

### Override testing with real 2025 DSL Orange data
```bash
DRY_RUN=1 OVERRIDE_GAMEPK=811804 python gameday_dsl_orange.py
DRY_RUN=1 OVERRIDE_DATE=2025-07-18 python gameday_dsl_orange.py
```

### Repost / WPA debug
```bash
DRY_RUN=1 FORCE_REPOST=1 DEBUG_WPA=1 OVERRIDE_GAMEPK=811804 python gameday_dsl_orange.py
```

### Daily recap
```bash
python gameday_dsl_orange.py --recap
```

## Notes
- `state.json` stores transaction seen IDs and DSL game/recap dedupe state.
- `prospects.json` is editable and used to prioritize players in the DSL post text and image highlighting.
- Prospect schema:
  ```json
  {
    "updated": "YYYY-MM-DD",
    "prospects": [
      {"name": "Player Name", "priority": 1, "personId": 123456}
    ]
  }
  ```
- `player_cache.json` caches per-player metadata (pitch hand/position).
- `data/tango_we.json` is the vendored WE lookup table used for WPA key moments (no live win-probability endpoint calls).

### GitHub Actions (run now from GitHub web UI)
1. Go to **Actions** → **DSL Giants Orange Final/Suspended Box Score**.
2. Click **Run workflow**.
3. Optionally set `override_gamepk` or `override_date`.
4. For a real post now:
   - set `dry_run=false`
   - set `force_repost=true` if the game/status was already posted.

If overrides are blank on manual run, the workflow picks the most recent DSL Orange game in the lookback window.
