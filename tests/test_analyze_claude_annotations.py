"""Tests for analyze_claude_annotations stats + human-derived helpers.

Pure functions only (no Hub I/O): Cramér's V bounds, the human validation
verdict/majority logic, human vote majority, and the distribution table.

Run from the repo root::

    python -m unittest tests.test_analyze
"""

from __future__ import annotations

import math
import unittest

import pandas as pd

import analyze_claude_annotations as az
import data


class CramersV(unittest.TestCase):
    def test_independent_is_low(self):
        # alternating vs blocked => no association
        a = pd.Series(list("ABAB" * 25))
        b = pd.Series(list("XXYY" * 25))
        v, n = az.cramers_v(a, b)
        self.assertEqual(n, 100)
        self.assertLess(v, 0.25)

    def test_perfect_dependence_is_one(self):
        a = pd.Series(list("AABB" * 25))
        b = pd.Series(list("XXYY" * 25))  # B fully determined by A
        v, n = az.cramers_v(a, b)
        self.assertAlmostEqual(v, 1.0, places=6)

    def test_empty_pairs_returns_nan_zero(self):
        v, n = az.cramers_v(pd.Series(["", ""]), pd.Series(["", ""]))
        self.assertEqual(n, 0)
        self.assertTrue(math.isnan(v))


def _val_row(*slots):
    """Build a row with the three validation struct columns from (choice, user)."""
    row = {}
    for i, col in enumerate(data.VALIDATION_COLS):
        choice, user = slots[i] if i < len(slots) else ("", "")
        row[col] = {"choice": choice, "username": user}
    return pd.Series(row)


def _vote_row(*slots):
    row = {}
    for i, col in enumerate(data.VOTE_COLS):
        choice, user = slots[i] if i < len(slots) else ("", "")
        row[col] = {"choice": choice, "username": user}
    return pd.Series(row)


class HumanValidation(unittest.TestCase):
    def test_no_slots_filled(self):
        self.assertEqual(az._human_validation(_val_row()), (0, None, None))

    def test_all_accept_is_accept(self):
        n, verdict, maj = az._human_validation(
            _val_row(("knowledge", "u1"), ("preference", "u2"), ("knowledge", "u3"))
        )
        self.assertEqual((n, verdict, maj), (3, "accept", "knowledge"))

    def test_any_reject_is_reject(self):
        n, verdict, _ = az._human_validation(
            _val_row(("knowledge", "u1"), ("trivial", "u2"))
        )
        self.assertEqual((n, verdict), (2, "reject"))

    def test_partial_fill_counts_only_filled(self):
        n, verdict, maj = az._human_validation(_val_row(("preference", "u1")))
        self.assertEqual((n, verdict, maj), (1, "accept", "preference"))


class HumanVote(unittest.TestCase):
    def test_majority(self):
        self.assertEqual(
            az._human_vote(_vote_row(("a", "u1"), ("a", "u2"), ("b", "u3"))), (3, "a")
        )

    def test_none_when_empty(self):
        self.assertEqual(az._human_vote(_vote_row()), (0, None))


class DistTable(unittest.TestCase):
    def test_counts_and_pct_ignore_empty(self):
        s = pd.Series(["a", "a", "b", "", "  "])
        rows = az.dist_table(s)
        self.assertEqual(rows[0], ("a", 2, 200 / 3))
        self.assertEqual({r[0] for r in rows}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
