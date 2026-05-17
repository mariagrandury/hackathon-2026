"""Unit tests for pure helpers in ``app.py`` (importing Gradio is fine —
we don't build the demo here, just call thin utilities).

Covers formatting (``_fmt_score``), pass-mark math (``_pass_raw``,
``_test_max_possible``), translation lookup (``_t``), language resolution
(``_resolve_language``), the writing-tab default-system-prompt builder,
the read-only display merger (``_merged_prompt_display``), the
``_clear_other_radio`` mutual-exclusion helper, and the small
validation-choices builders that feed the radios.

Run from the repo root::

    python -m unittest tests.test_app_helpers
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import app
from data import ACCEPT_CHOICES, REJECT_CHOICES, TEST_PASS_THRESHOLD


# ---------------------------------------------------------------------------
# _fmt_score / _pass_raw / _test_max_possible
# ---------------------------------------------------------------------------


class FmtScore(unittest.TestCase):
    """Display: integer if whole, one-decimal otherwise (no trailing .0)."""

    def test_integer_drops_decimal(self):
        self.assertEqual(app._fmt_score(0), "0")
        self.assertEqual(app._fmt_score(12), "12")
        self.assertEqual(app._fmt_score(12.0), "12")

    def test_half_point_keeps_one_decimal(self):
        self.assertEqual(app._fmt_score(11.5), "11.5")
        self.assertEqual(app._fmt_score(0.5), "0.5")

    def test_negative_values(self):
        # Partial-credit scheme can produce negative half-points.
        self.assertEqual(app._fmt_score(-7), "-7")
        self.assertEqual(app._fmt_score(-0.5), "-0.5")


class PassRaw(unittest.TestCase):
    """``TEST_PASS_THRESHOLD`` is now stored as raw points, not a fraction.
    ``_pass_raw`` just round-trips it for the display layer."""

    def test_returns_threshold_unchanged(self):
        with patch("app.TEST_PASS_THRESHOLD", 12.0):
            self.assertEqual(app._pass_raw(16), 12.0)

    def test_works_for_half_point_thresholds(self):
        # The display formatter handles 11.5 nicely (see _fmt_score), so a
        # half-point threshold should round-trip untouched.
        with patch("app.TEST_PASS_THRESHOLD", 11.5):
            self.assertEqual(app._pass_raw(16), 11.5)

    def test_max_possible_ignored(self):
        # Today ``max_possible`` doesn't affect the return — it's kept in
        # the signature in case we want to clamp/scale later.
        with patch("app.TEST_PASS_THRESHOLD", 10.0):
            self.assertEqual(app._pass_raw(16), 10.0)
            self.assertEqual(app._pass_raw(20), 10.0)

    def test_current_threshold_is_12_raw(self):
        # Sanity: the actual current configuration. Locks the documented
        # "need 12 / 16" pass mark so a future threshold tweak that breaks
        # the display strings gets caught here.
        self.assertEqual(app._pass_raw(16), 12.0)
        self.assertEqual(TEST_PASS_THRESHOLD, 12.0)


class TestMaxPossible(unittest.TestCase):
    """``_test_max_possible(lang)`` = #classification + #MCQ in entry test."""

    def test_real_bank_returns_16(self):
        # The current Spanish bank: 14 classification + 2 MCQ.
        self.assertEqual(app._test_max_possible("es"), 16)

    def test_returns_int(self):
        self.assertIsInstance(app._test_max_possible("es"), int)


# ---------------------------------------------------------------------------
# _t (translations)
# ---------------------------------------------------------------------------


class Translations(unittest.TestCase):

    def test_known_language_returns_its_strings(self):
        self.assertIn("test_intro", app._t("es"))
        self.assertEqual(app._t("es")["tab_writing"], "Escribir prompts")

    def test_unknown_language_falls_back_to_default(self):
        # Default is "en"; an unknown code shouldn't KeyError.
        self.assertEqual(app._t("fr"), app._t(app.DEFAULT_LANG))

    def test_none_falls_back_to_default(self):
        self.assertEqual(app._t(None), app._t(app.DEFAULT_LANG))


# ---------------------------------------------------------------------------
# _resolve_language
# ---------------------------------------------------------------------------


_PARTS = pd.DataFrame([
    {"username": "alice", "language": "es", "country": "es"},
    {"username": "bruno", "language": "pt", "country": "br"},
    {"username": "weird", "language": "kr", "country": "kr"},  # unsupported lang
])


def _prof(username: str) -> SimpleNamespace:
    return SimpleNamespace(username=username)


class ResolveLanguage(unittest.TestCase):

    def test_logged_out_returns_default(self):
        self.assertEqual(app._resolve_language(None), app.DEFAULT_LANG)

    def test_known_user_returns_their_language(self):
        self.assertEqual(app._resolve_language(_prof("alice"), _PARTS), "es")
        self.assertEqual(app._resolve_language(_prof("bruno"), _PARTS), "pt")

    def test_unknown_user_falls_back_to_default(self):
        self.assertEqual(app._resolve_language(_prof("nobody"), _PARTS), app.DEFAULT_LANG)

    def test_user_with_unsupported_language_falls_back(self):
        # If a participant's ``language`` field isn't in the T table (e.g.
        # someone got registered with "kr"), serve them the default rather
        # than crashing on key lookup.
        self.assertEqual(app._resolve_language(_prof("weird"), _PARTS), app.DEFAULT_LANG)


# ---------------------------------------------------------------------------
# _default_system_prompt
# ---------------------------------------------------------------------------


class DefaultSystemPrompt(unittest.TestCase):

    def test_no_country_returns_empty(self):
        # Logged-out / non-participant fallback — caller substitutes the
        # generic placeholder string.
        self.assertEqual(app._default_system_prompt("es", None), "")
        self.assertEqual(app._default_system_prompt("es", ""), "")

    def test_known_country_interpolated_in_default_lang(self):
        out = app._default_system_prompt("es", "es")
        self.assertIn("España", out)
        # Template includes "responde en español" — sanity-check we got the
        # ES template, not EN.
        self.assertIn("español", out.lower())

    def test_unknown_country_uppercases_code(self):
        # Falls through ``country_display``'s upper() branch.
        out = app._default_system_prompt("en", "uk")
        self.assertIn("UK", out)


# ---------------------------------------------------------------------------
# _merged_prompt_display
# ---------------------------------------------------------------------------


class MergedPromptDisplay(unittest.TestCase):

    def test_no_system_prompt_returns_prompt_only(self):
        self.assertEqual(app._merged_prompt_display("en", "", "the prompt"), "the prompt")

    def test_both_empty_returns_empty(self):
        self.assertEqual(app._merged_prompt_display("en", "", ""), "")

    def test_joined_with_blank_line(self):
        self.assertEqual(
            app._merged_prompt_display("en", "be brief", "what time?"),
            "be brief\n\nwhat time?",
        )

    def test_whitespace_stripped_before_merging(self):
        self.assertEqual(
            app._merged_prompt_display("en", "  sys  ", "  p  "),
            "sys\n\np",
        )

    def test_whitespace_only_system_prompt_treated_as_empty(self):
        self.assertEqual(
            app._merged_prompt_display("en", "   ", "p"),
            "p",
        )


# ---------------------------------------------------------------------------
# _clear_other_radio
# ---------------------------------------------------------------------------


class ClearOtherRadio(unittest.TestCase):
    """Mutual-exclusion helper between the Reject and Accept radios.
    Returning a ``gr.update()`` no-op when the input cleared is what
    prevents an infinite cascade (each .change firing the other)."""

    def test_value_present_clears_other(self):
        # When this radio gets a value, the other one should be cleared.
        upd = app._clear_other_radio("knowledge")
        self.assertEqual(upd.get("value"), None)

    def test_none_input_is_noop(self):
        # ``gr.update()`` with no kwargs has no `value` key — that's the
        # noop signal that doesn't re-trigger the other radio's .change.
        upd = app._clear_other_radio(None)
        self.assertNotIn("value", upd)

    def test_empty_string_input_is_noop(self):
        # Falsy string → treated the same as None.
        upd = app._clear_other_radio("")
        self.assertNotIn("value", upd)


# ---------------------------------------------------------------------------
# show_user
# ---------------------------------------------------------------------------


class ShowUser(unittest.TestCase):

    def test_logged_out(self):
        out = app.show_user("en", None)
        self.assertIn("Not logged in", out)

    def test_logged_in(self):
        out = app.show_user("en", _prof("alice"))
        self.assertIn("alice", out)


# ---------------------------------------------------------------------------
# _read_guidelines
# ---------------------------------------------------------------------------


class ReadGuidelines(unittest.TestCase):

    def test_existing_language(self):
        # Spanish is the source-of-truth file and definitely exists.
        out = app._read_guidelines("es")
        self.assertGreater(len(out), 100)

    def test_missing_language_returns_translation_fallback(self):
        # An unknown lang code can't have a guidelines file; should fall
        # back to the ``guidelines_missing`` translated string.
        out = app._read_guidelines("xx")
        self.assertEqual(out, app._t("xx")["guidelines_missing"])


# ---------------------------------------------------------------------------
# Validation-choice builders
# ---------------------------------------------------------------------------


class ValidationChoiceBuilders(unittest.TestCase):

    def test_reject_choices_shape(self):
        out = app._validation_reject_choices("en")
        self.assertEqual(len(out), len(REJECT_CHOICES))
        # (label, value) tuples; values match the canonical bucket keys.
        values = [v for _, v in out]
        self.assertEqual(values, list(REJECT_CHOICES))
        # Labels are translated.
        for label, _ in out:
            self.assertIsInstance(label, str)
            self.assertTrue(label)

    def test_accept_choices_shape(self):
        out = app._validation_accept_choices("en")
        self.assertEqual(len(out), len(ACCEPT_CHOICES))
        values = [v for _, v in out]
        self.assertEqual(values, list(ACCEPT_CHOICES))

    def test_localised_labels_differ_across_languages(self):
        # Same value, different label — confirms the localisation path runs.
        en_labels = [l for l, _ in app._validation_reject_choices("en")]
        es_labels = [l for l, _ in app._validation_reject_choices("es")]
        self.assertNotEqual(en_labels, es_labels)


if __name__ == "__main__":
    unittest.main()
