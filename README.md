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
