"""Unit tests for pure helpers in ``data.py`` (no HF, no Gradio).

Covers JSON parsing (``parse_test_score``), score lookups (``best_test_score``),
country display, the row predicates (``is_fully_validated`` / ``has_answers``
and the vectorised ``_fully_validated_mask``), and the leaderboard
aggregations (``user_stats`` / ``country_counts`` / ``ranking_df`` /
``all_known_usernames``). All synthetic data — no Hub access.

Run from the repo root::

    python -m unittest tests.test_data_helpers
"""

from __future__ import annotations

import unittest

import pandas as pd

from data import (
    ACCEPT_CHOICES,
    EXCLUDED_USERNAMES,
    REJECT_CHOICES,
    _fully_validated_mask,
    all_known_usernames,
    best_test_score,
    country_counts,
    country_display,
    has_answers,
    is_fully_validated,
    parse_test_responses,
    parse_test_score,
    ranking_df,
    user_stats,
)

EMPTY_SLOT = {"choice": "", "username": ""}


def _val(user: str, choice: str = "knowledge") -> dict:
    return {"choice": choice, "username": user}


def _vote(user: str, choice: str = "a") -> dict:
    return {"choice": choice, "username": user}


def _prompt_row(
    *,
    id: int = 1,
    username: str = "u",
    country: str = "es",
    val: tuple[dict, dict, dict] = (EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT),
    vote: tuple[dict, dict, dict] = (EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT),
    answer_a: str = "",
    answer_b: str = "",
) -> dict:
    """Minimal prompt-table row matching the shape ``data.PROMPTS_FEATURES``
    expects (only the fields the helpers under test actually read)."""
    return {
        "id": id,
        "username": username,
        "country": country,
        "prompt_validation_1": val[0],
        "prompt_validation_2": val[1],
        "prompt_validation_3": val[2],
        "answer_a": answer_a,
        "answer_b": answer_b,
        "answer_chosen_1": vote[0],
        "answer_chosen_2": vote[1],
        "answer_chosen_3": vote[2],
    }


# ---------------------------------------------------------------------------
# parse_test_score
# ---------------------------------------------------------------------------


class ParseTestScore(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(parse_test_score(None), {})

    def test_empty_string_returns_empty(self):
        self.assertEqual(parse_test_score(""), {})

    def test_empty_object_returns_empty(self):
        self.assertEqual(parse_test_score("{}"), {})

    def test_single_attempt(self):
        self.assertEqual(parse_test_score('{"1": 0.85}'), {"1": 0.85})

    def test_multiple_attempts(self):
        self.assertEqual(
            parse_test_score('{"1": 1.0, "2": 0.5, "3": -0.25}'),
            {"1": 1.0, "2": 0.5, "3": -0.25},
        )

    def test_invalid_json_returns_empty(self):
        # Corrupt cell shouldn't crash a page load.
        self.assertEqual(parse_test_score("not json"), {})
        self.assertEqual(parse_test_score("{1: 0.5"), {})

    def test_non_dict_returns_empty(self):
        # Defensive: a list or scalar in the cell is treated as "no scores".
        self.assertEqual(parse_test_score("[1, 2, 3]"), {})
        self.assertEqual(parse_test_score("42"), {})

    def test_keys_coerced_to_str_values_to_float(self):
        # JSON object keys are always strings already; integer values get
        # promoted to floats so callers don't have to special-case.
        result = parse_test_score('{"1": 1}')
        self.assertEqual(result, {"1": 1.0})
        self.assertIsInstance(next(iter(result.values())), float)


# ---------------------------------------------------------------------------
# parse_test_responses
# ---------------------------------------------------------------------------


class ParseTestResponses(unittest.TestCase):

    def test_none_and_empty(self):
        self.assertEqual(parse_test_responses(None), {})
        self.assertEqual(parse_test_responses(""), {})
        self.assertEqual(parse_test_responses("{}"), {})

    def test_single_attempt(self):
        self.assertEqual(
            parse_test_responses('{"1": {"CLF_01": "trivial", "MCQ_01": "a"}}'),
            {"1": {"CLF_01": "trivial", "MCQ_01": "a"}},
        )

    def test_multiple_attempts(self):
        self.assertEqual(
            parse_test_responses(
                '{"1": {"CLF_01": "trivial"}, "2": {"CLF_01": "knowledge"}}'
            ),
            {"1": {"CLF_01": "trivial"}, "2": {"CLF_01": "knowledge"}},
        )

    def test_invalid_json_returns_empty(self):
        self.assertEqual(parse_test_responses("not json"), {})
        self.assertEqual(parse_test_responses("{1: trivial"), {})

    def test_non_dict_returns_empty(self):
        self.assertEqual(parse_test_responses("[1, 2, 3]"), {})
        self.assertEqual(parse_test_responses('"hello"'), {})

    def test_skips_non_dict_inner_values(self):
        # Defensive: if one attempt's value is somehow a string/list instead
        # of a dict (corrupt cell), skip it rather than crash. The other
        # attempts in the same cell should still parse.
        self.assertEqual(
            parse_test_responses('{"1": {"CLF_01": "a"}, "2": "broken"}'),
            {"1": {"CLF_01": "a"}},
        )

    def test_coerces_keys_and_values_to_str(self):
        # JSON dict keys are always strings; values should be too (so we
        # never accidentally store an int / None in the per-question map).
        result = parse_test_responses('{"1": {"CLF_01": 42}}')
        self.assertEqual(result, {"1": {"CLF_01": "42"}})


# ---------------------------------------------------------------------------
# best_test_score
# ---------------------------------------------------------------------------


def _participants_with_scores(scores_by_user: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"username": u, "language": "es", "country": "es",
             "test_score": s}
            for u, s in scores_by_user.items()
        ]
    )


class BestTestScore(unittest.TestCase):

    def test_none_username_returns_zero(self):
        # No HF call needed — short-circuit on falsy username.
        self.assertEqual(best_test_score(None), 0.0)

    def test_empty_username_returns_zero(self):
        self.assertEqual(best_test_score(""), 0.0)

    def test_user_not_in_df_returns_zero(self):
        df = _participants_with_scores({"alice": "{}"})
        self.assertEqual(best_test_score("bob", df), 0.0)

    def test_user_with_no_attempts_returns_zero(self):
        df = _participants_with_scores({"alice": "{}"})
        self.assertEqual(best_test_score("alice", df), 0.0)

    def test_user_with_single_attempt(self):
        df = _participants_with_scores({"alice": '{"1": 0.875}'})
        self.assertEqual(best_test_score("alice", df), 0.875)

    def test_max_across_attempts(self):
        # The whole point: a later worse attempt mustn't lock out a user who
        # already passed. ``best_test_score`` returns the max.
        df = _participants_with_scores({"alice": '{"1": 0.5, "2": 0.95, "3": 0.3}'})
        self.assertEqual(best_test_score("alice", df), 0.95)

    def test_negative_scores_handled(self):
        # Partial-credit grading can produce negative fractions; best across
        # all-negative attempts is still the max (least negative).
        df = _participants_with_scores({"alice": '{"1": -0.5, "2": -0.1}'})
        self.assertEqual(best_test_score("alice", df), -0.1)


# ---------------------------------------------------------------------------
# country_display
# ---------------------------------------------------------------------------


class CountryDisplay(unittest.TestCase):

    def test_none_shows_question_mark(self):
        self.assertEqual(country_display(None), "?")

    def test_empty_shows_question_mark(self):
        self.assertEqual(country_display(""), "?")

    def test_known_code_shows_display_name(self):
        self.assertEqual(country_display("es"), "España")
        self.assertEqual(country_display("br"), "Brasil")
        self.assertEqual(country_display("do"), "República Dominicana")

    def test_unknown_code_uppercases(self):
        # Fallback so a new country shows up legibly until added to the map.
        self.assertEqual(country_display("xx"), "XX")
        self.assertEqual(country_display("uk"), "UK")


# ---------------------------------------------------------------------------
# is_fully_validated / has_answers
# ---------------------------------------------------------------------------


class IsFullyValidated(unittest.TestCase):

    def test_all_three_accept_returns_true(self):
        row = _prompt_row(val=(_val("a"), _val("b", "preference"), _val("c", "dynamics")))
        self.assertTrue(is_fully_validated(row))

    def test_any_empty_slot_returns_false(self):
        row = _prompt_row(val=(_val("a"), EMPTY_SLOT, _val("c")))
        self.assertFalse(is_fully_validated(row))

    def test_any_reject_bucket_returns_false(self):
        for reject in REJECT_CHOICES:
            with self.subTest(reject=reject):
                row = _prompt_row(val=(_val("a"), _val("b", reject), _val("c")))
                self.assertFalse(is_fully_validated(row))

    def test_all_reject_returns_false(self):
        row = _prompt_row(
            val=(_val("a", "trivial"), _val("b", "stereotype"), _val("c", "unrelated"))
        )
        self.assertFalse(is_fully_validated(row))


class HasAnswers(unittest.TestCase):

    def test_both_filled_returns_true(self):
        row = _prompt_row(answer_a="A", answer_b="B")
        self.assertTrue(has_answers(row))

    def test_a_missing_returns_false(self):
        self.assertFalse(has_answers(_prompt_row(answer_a="", answer_b="B")))

    def test_b_missing_returns_false(self):
        self.assertFalse(has_answers(_prompt_row(answer_a="A", answer_b="")))

    def test_whitespace_treated_as_empty(self):
        self.assertFalse(has_answers(_prompt_row(answer_a="   ", answer_b="B")))
        self.assertFalse(has_answers(_prompt_row(answer_a="A", answer_b="\n")))

    def test_missing_keys_treated_as_empty(self):
        # ``has_answers`` uses ``row.get`` for resilience — a row that
        # somehow lost an answer column shouldn't blow up.
        self.assertFalse(has_answers({}))


# ---------------------------------------------------------------------------
# _fully_validated_mask (vectorised)
# ---------------------------------------------------------------------------


class FullyValidatedMask(unittest.TestCase):

    def test_matches_row_predicate(self):
        # Mix of rows: 1 fully validated, 1 partly empty, 1 with a reject.
        rows = [
            _prompt_row(id=1, val=(_val("a"), _val("b", "preference"), _val("c", "dynamics"))),
            _prompt_row(id=2, val=(_val("a"), EMPTY_SLOT, _val("c"))),
            _prompt_row(id=3, val=(_val("a"), _val("b", "trivial"), _val("c"))),
        ]
        df = pd.DataFrame(rows)
        mask = _fully_validated_mask(df).tolist()
        expected = [is_fully_validated(r) for r in rows]
        self.assertEqual(mask, expected)
        self.assertEqual(mask, [True, False, False])


# ---------------------------------------------------------------------------
# user_stats
# ---------------------------------------------------------------------------


class UserStats(unittest.TestCase):

    def test_empty_username_returns_zeros(self):
        df = pd.DataFrame([_prompt_row(username="alice")])
        self.assertEqual(user_stats("", df), {"sent": 0, "validated": 0, "voted": 0})

    def test_empty_df_returns_zeros(self):
        self.assertEqual(
            user_stats("alice", pd.DataFrame()),
            {"sent": 0, "validated": 0, "voted": 0},
        )

    def test_counts_across_columns(self):
        # alice authored 2, validated on 3 different rows (across both
        # validator-column positions), voted once.
        rows = [
            _prompt_row(id=1, username="alice"),
            _prompt_row(id=2, username="alice", val=(_val("bob"), EMPTY_SLOT, EMPTY_SLOT)),
            _prompt_row(id=3, username="bob", val=(_val("alice"), EMPTY_SLOT, EMPTY_SLOT)),
            _prompt_row(
                id=4,
                username="bob",
                val=(_val("alice"), _val("carla"), EMPTY_SLOT),
                vote=(_vote("alice"), EMPTY_SLOT, EMPTY_SLOT),
            ),
            _prompt_row(
                id=5,
                username="bob",
                val=(EMPTY_SLOT, EMPTY_SLOT, _val("alice")),
            ),
        ]
        df = pd.DataFrame(rows)
        stats = user_stats("alice", df)
        self.assertEqual(stats, {"sent": 2, "validated": 3, "voted": 1})

    def test_unknown_user_returns_zeros(self):
        df = pd.DataFrame([_prompt_row(username="alice")])
        self.assertEqual(
            user_stats("nobody", df),
            {"sent": 0, "validated": 0, "voted": 0},
        )


# ---------------------------------------------------------------------------
# country_counts
# ---------------------------------------------------------------------------


class CountryCounts(unittest.TestCase):

    def test_empty_df_returns_empty(self):
        result = country_counts(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertListEqual(
            list(result.columns), ["country", "fully_validated", "pending"]
        )

    def test_splits_validated_vs_pending(self):
        rows = [
            # 2 ES rows: 1 fully validated, 1 pending.
            _prompt_row(id=1, country="es",
                        val=(_val("a"), _val("b", "preference"), _val("c", "dynamics"))),
            _prompt_row(id=2, country="es"),
            # 1 BR row, fully validated.
            _prompt_row(id=3, country="br",
                        val=(_val("a"), _val("b"), _val("c"))),
        ]
        result = country_counts(pd.DataFrame(rows)).set_index("country")
        self.assertEqual(int(result.at["es", "fully_validated"]), 1)
        self.assertEqual(int(result.at["es", "pending"]), 1)
        self.assertEqual(int(result.at["br", "fully_validated"]), 1)
        self.assertEqual(int(result.at["br", "pending"]), 0)

    def test_drops_rows_with_missing_country(self):
        rows = [
            _prompt_row(id=1, country="es"),
            _prompt_row(id=2, country=None),  # NaN-equivalent — dropped.
        ]
        result = country_counts(pd.DataFrame(rows))
        self.assertNotIn(None, result["country"].tolist())
        self.assertEqual(int(result.at[result.index[0], "pending"]), 1)


# ---------------------------------------------------------------------------
# ranking_df + all_known_usernames
# ---------------------------------------------------------------------------


class Ranking(unittest.TestCase):

    def _sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                # author: alice / bob / v0
                # validators: bob, carla (alice never validates)
                # voters: alice
                _prompt_row(id=1, username="alice",
                            val=(_val("bob"), _val("carla"), EMPTY_SLOT)),
                _prompt_row(id=2, username="bob",
                            val=(EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT),
                            vote=(_vote("alice"), EMPTY_SLOT, EMPTY_SLOT)),
                _prompt_row(id=3, username="v0",
                            val=(_val("bob"), EMPTY_SLOT, EMPTY_SLOT)),
            ]
        )

    def test_ranking_excludes_v0(self):
        result = ranking_df(self._sample())
        self.assertNotIn("v0", result["username"].tolist())

    def test_ranking_sorts_by_prompts_sent_then_validated_voted(self):
        result = ranking_df(self._sample())
        # alice authored 1, bob authored 1 — tie on sent. bob validated 2
        # (rows 1 + 3), so should rank ahead of alice on the tiebreak.
        self.assertEqual(result.iloc[0]["username"], "bob")
        self.assertEqual(int(result.iloc[0]["prompts validated"]), 2)
        self.assertEqual(int(result.iloc[1]["prompts sent"]), 1)

    def test_ranking_includes_validator_only_users(self):
        # carla never authored or voted, only validated. Should still appear.
        result = ranking_df(self._sample())
        self.assertIn("carla", result["username"].tolist())

    def test_ranking_empty_df(self):
        result = ranking_df(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertListEqual(
            list(result.columns),
            ["username", "prompts sent", "prompts validated", "answers voted"],
        )


class AllKnownUsernames(unittest.TestCase):

    def test_empty_df(self):
        self.assertEqual(all_known_usernames(pd.DataFrame()), [])

    def test_dedupes_across_columns(self):
        # alice appears as both author and validator; should appear once.
        df = pd.DataFrame([
            _prompt_row(id=1, username="alice",
                        val=(_val("alice"), _val("bob"), EMPTY_SLOT)),
        ])
        self.assertEqual(all_known_usernames(df), ["alice", "bob"])

    def test_excludes_v0_sentinel(self):
        df = pd.DataFrame([
            _prompt_row(id=1, username="v0", val=(_val("bob"), EMPTY_SLOT, EMPTY_SLOT)),
            _prompt_row(id=2, username="alice"),
        ])
        result = all_known_usernames(df)
        self.assertNotIn("v0", result)
        # And the excluded set is the source of truth — this also catches
        # accidental new sentinels being added without filtering.
        self.assertTrue("v0" in EXCLUDED_USERNAMES)

    def test_drops_empty_username_strings(self):
        # Empty-string username slots (e.g. unfilled vote rows) shouldn't
        # leak into the leaderboard.
        df = pd.DataFrame([
            _prompt_row(id=1, username="alice",
                        vote=(_vote(""), EMPTY_SLOT, EMPTY_SLOT)),
        ])
        self.assertNotIn("", all_known_usernames(df))


if __name__ == "__main__":
    unittest.main()
