"""Unit tests for the entry-test grading in ``test_data``.

Per-question scoring (``_score_classification``, ``_score_mcq``) and the
``grade`` composition are pure functions, so they're exhaustively covered
against synthetic question banks (patched in via ``unittest.mock``). The
``BankLoaders`` section does light integration with the real
``data/test-2026.json`` to lock the entry-test / hidden-test invariants.

Run from the repo root::

    python -m unittest tests.test_grading       # one file
    python -m unittest discover tests           # everything in tests/
"""

from __future__ import annotations

import unittest
from collections import Counter
from unittest.mock import patch

from data import ACCEPT_CHOICES, REJECT_CHOICES, VALIDATION_CHOICES
from test_data import (
    TEST_QUESTIONS_PER_CATEGORY,
    _score_classification,
    _score_mcq,
    grade,
    load_hidden_questions,
    load_mcq_questions,
    load_questions,
)


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------


class ClassificationScoring(unittest.TestCase):
    """``_score_classification`` per-question scoring rules:
    +1 exact / +0.5 same side, wrong bucket / -0.5 wrong side / 0 unanswered."""

    def test_unanswered_is_zero(self):
        self.assertEqual(
            _score_classification({"correct_key": "trivial"}, None), 0.0
        )

    def test_exact_match_is_one(self):
        # Every canonical bucket scores +1 when picked exactly.
        for key in VALIDATION_CHOICES:
            with self.subTest(key=key):
                self.assertEqual(
                    _score_classification({"correct_key": key}, key), 1.0
                )

    def test_same_side_wrong_bucket_reject(self):
        # correct = trivial (reject); pick the other two reject buckets → +0.5
        q = {"correct_key": "trivial"}
        for picked in ("stereotype", "unrelated"):
            with self.subTest(picked=picked):
                self.assertEqual(_score_classification(q, picked), 0.5)

    def test_same_side_wrong_bucket_accept(self):
        # correct = knowledge; pick the other three accept buckets → +0.5
        q = {"correct_key": "knowledge"}
        for picked in ("preference", "dynamics", "bias_probe"):
            with self.subTest(picked=picked):
                self.assertEqual(_score_classification(q, picked), 0.5)

    def test_wrong_side_reject_to_accept(self):
        q = {"correct_key": "trivial"}
        for picked in ACCEPT_CHOICES:
            with self.subTest(picked=picked):
                self.assertEqual(_score_classification(q, picked), -0.5)

    def test_wrong_side_accept_to_reject(self):
        q = {"correct_key": "knowledge"}
        for picked in REJECT_CHOICES:
            with self.subTest(picked=picked):
                self.assertEqual(_score_classification(q, picked), -0.5)


class MCQScoring(unittest.TestCase):
    """``_score_mcq`` per-question scoring rules: +1 correct / -1 wrong / 0 blank."""

    def test_unanswered_is_zero(self):
        self.assertEqual(_score_mcq({"correct": "Option A"}, None), 0.0)

    def test_exact_match_is_one(self):
        self.assertEqual(_score_mcq({"correct": "Option A"}, "Option A"), 1.0)

    def test_wrong_choice_is_minus_one(self):
        q = {"correct": "Option A"}
        for picked in ("Option B", "Option C", "", "random"):
            with self.subTest(picked=picked):
                self.assertEqual(_score_mcq(q, picked), -1.0)


# ---------------------------------------------------------------------------
# grade(): composition + routing
# ---------------------------------------------------------------------------

# Synthetic banks let the routing tests assert exact raw scores without
# coupling to whatever's in data/test-2026.json today.
_FAKE_CLASSIF = [
    {"id": "C1", "correct_key": "trivial"},    # reject side
    {"id": "C2", "correct_key": "knowledge"},  # accept side
]
_FAKE_MCQ = [{"id": "M1", "correct": "right", "options": ["right", "wrong"]}]


def _patch_banks(classif=None, mcq=None):
    """Patch ``load_questions`` and ``load_mcq_questions`` on the
    ``test_data`` module so ``grade`` sees the supplied synthetic bank."""
    return patch.multiple(
        "test_data",
        load_questions=lambda lang: list(
            classif if classif is not None else _FAKE_CLASSIF
        ),
        load_mcq_questions=lambda lang: list(
            mcq if mcq is not None else _FAKE_MCQ
        ),
    )


class GradeRouting(unittest.TestCase):

    def test_perfect_score(self):
        with _patch_banks():
            score, raw, max_possible = grade(
                [("C1", "trivial"), ("C2", "knowledge"), ("M1", "right")],
                "es",
            )
        self.assertEqual(raw, 3.0)
        self.assertEqual(max_possible, 3.0)
        self.assertEqual(score, 1.0)

    def test_all_wrong(self):
        # Wrong-side classification (-0.5 each) + wrong MCQ (-1) = -2
        with _patch_banks():
            score, raw, max_possible = grade(
                [("C1", "knowledge"), ("C2", "trivial"), ("M1", "wrong")],
                "es",
            )
        self.assertEqual(raw, -2.0)
        self.assertEqual(max_possible, 3.0)
        self.assertLess(score, 0)  # fractions can go negative

    def test_unanswered_counts_as_zero(self):
        # Unanswered pairs don't add to raw and don't penalize.
        with _patch_banks():
            score, raw, max_possible = grade(
                [("C1", None), ("C2", None), ("M1", None)],
                "es",
            )
        self.assertEqual(raw, 0.0)
        self.assertEqual(max_possible, 3.0)
        self.assertEqual(score, 0.0)

    def test_unknown_id_silently_skipped(self):
        # An id not in either bank shouldn't crash or affect the raw score.
        with _patch_banks():
            _, raw, _ = grade(
                [("does_not_exist", "trivial"), ("C1", "trivial")], "es"
            )
        self.assertEqual(raw, 1.0)

    def test_empty_bank_returns_zero_tuple(self):
        # Documented shortcut: no questions → (0, 0, 0), no division by zero.
        with _patch_banks(classif=[], mcq=[]):
            self.assertEqual(grade([("anything", "trivial")], "es"), (0.0, 0.0, 0.0))

    def test_partial_credit_composition(self):
        # 1 exact (+1) + 1 same-side-wrong (+0.5) + 1 MCQ wrong (-1) = 0.5
        with _patch_banks():
            _, raw, _ = grade(
                [("C1", "trivial"), ("C2", "preference"), ("M1", "wrong")],
                "es",
            )
        self.assertEqual(raw, 0.5)

    def test_classification_and_mcq_routed_correctly(self):
        # Same chosen value passed as both a classification answer (bucket
        # key) and an MCQ answer (option string) — verify each is graded
        # by the right scorer.
        classif = [{"id": "C1", "correct_key": "trivial"}]
        mcq = [{"id": "M1", "correct": "trivial", "options": ["trivial", "x"]}]
        with _patch_banks(classif=classif, mcq=mcq):
            _, raw, _ = grade([("C1", "trivial"), ("M1", "trivial")], "es")
        # Classification: +1 (exact). MCQ: +1 (string match). Total 2.
        self.assertEqual(raw, 2.0)


# ---------------------------------------------------------------------------
# Bank loaders — light integration with the real JSON
# ---------------------------------------------------------------------------


class BankLoaders(unittest.TestCase):
    """Locks the invariants the rest of the app relies on:
    - entry test has TEST_QUESTIONS_PER_CATEGORY items per bucket,
    - entry and hidden banks are disjoint,
    - MCQ items have ``options`` + ``correct`` and ``correct`` ∈ ``options``.
    """

    def test_entry_test_quota_per_category(self):
        counts = Counter(q["correct_key"] for q in load_questions("es"))
        for key in VALIDATION_CHOICES:
            with self.subTest(category=key):
                self.assertEqual(
                    counts.get(key, 0),
                    TEST_QUESTIONS_PER_CATEGORY,
                    msg=f"expected {TEST_QUESTIONS_PER_CATEGORY} {key} questions",
                )

    def test_entry_in_canonical_order(self):
        # Entries grouped by category in VALIDATION_CHOICES order (reject
        # buckets first, then accept buckets).
        prev_rank = -1
        for q in load_questions("es"):
            rank = VALIDATION_CHOICES.index(q["correct_key"])
            self.assertGreaterEqual(rank, prev_rank)
            prev_rank = rank

    def test_entry_and_hidden_are_disjoint(self):
        entry_ids = {q["id"] for q in load_questions("es")}
        hidden_ids = {q["id"] for q in load_hidden_questions("es")}
        self.assertFalse(
            entry_ids & hidden_ids,
            msg=f"overlap: {entry_ids & hidden_ids}",
        )

    def test_mcq_shape(self):
        mcq = load_mcq_questions("es")
        self.assertGreater(len(mcq), 0)
        for q in mcq:
            with self.subTest(id=q.get("id")):
                self.assertIn("options", q)
                self.assertIn("correct", q)
                # The correct answer must be one of the options — otherwise
                # _score_mcq would never award +1, no matter the user input.
                self.assertIn(q["correct"], q["options"])


if __name__ == "__main__":
    unittest.main()
