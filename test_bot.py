import unittest
from unittest.mock import patch

import bot


class BotTests(unittest.TestCase):
    def event(self, **overrides):
        data = dict(
            id=1,
            sort_date="2026-08-18",
            person_id=123,
            person_name="Test Player",
            position="SS",
            event_type="assignment",
            from_id=bot.SAN_JOSE,
            from_name="San Jose Giants",
            to_id=bot.EUGENE,
            to_name="Eugene Emeralds",
            display_team_id=bot.EUGENE,
            header="Eugene",
            description="SS Test Player assigned to Eugene Emeralds from San Jose Giants.",
            event_key="key",
        )
        data.update(overrides)
        return bot.TxnEvent(**data)

    def test_level_direction_does_not_infer_promotion(self):
        event = self.event()
        text = bot.level_change_text(event)
        self.assertIn("SS Test Player → High-A Eugene", text)
        self.assertIn("From Low-A San Jose", text)
        self.assertNotIn("promotion", text.lower())
        self.assertNotIn("promoted", text.lower())

    def test_event_classifier_preserves_explicit_transaction_types(self):
        self.assertEqual(bot.classify_event("Sacramento River Cats recalled RHP A from Richmond."), "recalled")
        self.assertEqual(bot.classify_event("San Jose Giants placed OF B on the 7-day injured list."), "il_placed")
        self.assertEqual(bot.classify_event("Richmond Flying Squirrels released RHP C."), "released")
        self.assertEqual(bot.classify_event("SS D assigned to Eugene Emeralds from San Jose Giants."), "assignment")

    def test_event_key_dedupes_same_normalized_event(self):
        a = bot.make_event_key("2026-08-18", 123, "assignment", bot.SAN_JOSE, bot.EUGENE, "SS X assigned to Eugene.")
        b = bot.make_event_key("2026-08-18T12:30:00Z", 123, "assignment", bot.SAN_JOSE, bot.EUGENE, "SS   X assigned to Eugene .")
        self.assertEqual(a, b)

    def test_affiliate_prefix_is_removed_from_plain_transaction(self):
        event = self.event(
            event_type="il_placed",
            from_id=bot.RICHMOND,
            from_name="Richmond Flying Squirrels",
            to_id=None,
            to_name=None,
            display_team_id=bot.RICHMOND,
            header="Richmond",
            description="Richmond Flying Squirrels placed LHP Tyler Example on the 7-day injured list.",
            position="LHP",
            person_name="Tyler Example",
        )
        self.assertEqual(bot.make_compact_line(event), "placed LHP Tyler Example on the 7-day IL.")

    def test_hitter_stats_context_uses_season_and_meaningful_last_14(self):
        event = self.event()
        season = {"plateAppearances": 250, "avg": ".282", "obp": ".350", "slg": ".462", "homeRuns": 9}
        recent = {"plateAppearances": 55, "avg": ".347", "obp": ".418", "slg": ".612", "homeRuns": 3}
        with patch("bot.fetch_date_range_stats", side_effect=[season, recent]):
            bot.attach_stats_context(None, event)
        self.assertEqual(event.stats_lines[0], "SJ season: .282/.350/.462 in 250 PA, 9 HR")
        self.assertEqual(event.stats_lines[1], "Last 14d: .347/.418/.612 in 55 PA")

    def test_hitter_recent_split_is_omitted_below_sample_threshold(self):
        event = self.event()
        season = {"plateAppearances": 100, "avg": ".250", "obp": ".330", "slg": ".410", "homeRuns": 4}
        recent = {"plateAppearances": 19, "avg": ".400", "obp": ".500", "slg": ".700", "homeRuns": 2}
        with patch("bot.fetch_date_range_stats", side_effect=[season, recent]):
            bot.attach_stats_context(None, event)
        self.assertEqual(len(event.stats_lines), 1)
        self.assertTrue(event.stats_lines[0].startswith("SJ season:"))

    def test_pitcher_stats_context_uses_k_and_bb_rates(self):
        event = self.event(position="RHP", person_name="Pitcher Example")
        season = {
            "inningsPitched": "72.1",
            "era": "3.48",
            "strikeOuts": 80,
            "baseOnBalls": 24,
            "battersFaced": 300,
        }
        recent = {
            "inningsPitched": "12.0",
            "era": "2.25",
            "strikeOuts": 17,
            "baseOnBalls": 3,
            "battersFaced": 50,
        }
        with patch("bot.fetch_date_range_stats", side_effect=[season, recent]):
            bot.attach_stats_context(None, event)
        self.assertEqual(event.stats_lines[0], "SJ season: 72.1 IP, 3.48 ERA · 26.7% K, 8.0% BB")
        self.assertEqual(event.stats_lines[1], "Last 14d: 12.0 IP, 2.25 ERA · 34.0% K, 6.0% BB")

    def test_enriched_level_change_gets_own_post(self):
        event = self.event(stats_lines=["SJ season: .282/.350/.462 in 250 PA, 9 HR"])
        posts = bot.build_posts([event])
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].event_ids, [1])
        self.assertIn("SJ season:", posts[0].text)

    def test_grouped_post_continuation_keeps_header(self):
        events = []
        for i in range(8):
            events.append(
                self.event(
                    id=i + 1,
                    event_type="released",
                    from_id=bot.RICHMOND,
                    from_name="Richmond Flying Squirrels",
                    to_id=None,
                    to_name=None,
                    display_team_id=bot.RICHMOND,
                    header="Richmond",
                    person_name=f"Player {i}",
                    position="RHP",
                    description=f"Richmond Flying Squirrels released RHP Player {i} after a roster transaction.",
                    stats_lines=[],
                )
            )
        posts = bot.build_posts(events)
        self.assertGreaterEqual(len(posts), 2)
        self.assertTrue(posts[1].text.startswith("Richmond (cont.)"))
        self.assertTrue(all(len(p.text) <= bot.MAX_CHARS for p in posts))


if __name__ == "__main__":
    unittest.main()
