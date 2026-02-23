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
OVERRIDE_GAMEPK=811804 python gameday_dsl_orange.py
OVERRIDE_DATE=2025-07-18 python gameday_dsl_orange.py
```

### Daily recap
```bash
python gameday_dsl_orange.py --recap
```

## Notes
- `state.json` stores transaction seen IDs and DSL game/recap dedupe state.
- `prospects.json` is editable and used to prioritize top prospects in summaries.
- If `prospects.json` is stale (older than 45 days), the bot logs a warning.


### GitHub Actions (run now from GitHub web UI)
1. Go to **Actions** → **DSL Giants Orange Final/Suspended Box Score**.
2. Click **Run workflow**.
3. Set either:
   - `override_gamepk=811804` (recommended), or
   - `override_date=2025-07-18`
4. For a real post now:
   - set `dry_run=false`
   - set `force_repost=true` if the game was already posted before.

> `force_repost=true` bypasses state dedupe for that run only.


If you do not see input fields in GitHub UI, you are running an old workflow revision/branch. Pick the branch containing this file update first.
