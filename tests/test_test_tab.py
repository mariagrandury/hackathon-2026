"""Integration tests for the entry-test flow in ``app``.

Drives ``load_test`` / ``submit_test`` with the real Spanish question bank
plus a fake ``record_test_attempt`` so we never hit the Hub. Verifies:

  - shape of ``load_test`` outputs (state + flat block updates),
  - "already passed" shortcut emits the celebratory message and hides
    submit/retake,
  - ``submit_test`` grading routes correctly: perfect → passed + reveals
    gated tabs, mixed → failed + tabs untouched, all-blank → "answer all
    questions" + tabs untouched, missing-participant → not-participant
    error,
  - logged-out short-circuit before grading.

Run from the repo root::

    python -m unittest tests.test_test_tab
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import app
from data import REJECT_CHOICES
from test_data import load_mcq_questions, load_questions


def _prof(username: str) -> SimpleNamespace:
    return SimpleNamespace(username=username)


# Bypass the participants-table lookup in ``best_test_score`` by always
# passing in a synthetic df. The "default" df below makes every user start
# fresh; individual tests build their own when they need a passing user.
_PARTS_FRESH = pd.DataFrame(
    [
        {"username": "alice", "language": "es", "country": "es",
         "gmail": "", "test_score": "{}"},
    ]
)


def _parts_with_score(username: str, score: float) -> pd.DataFrame:
    import json
    return pd.DataFrame(
        [
            {"username": username, "language": "es", "country": "es",
             "gmail": "", "test_score": json.dumps({"1": score})},
        ]
    )


# Capture record_test_attempt calls; lets us assert that submit_test
# persisted (or skipped) the score per branch.
class _RecordSpy:
    def __init__(self):
        self.calls: list[tuple[str, float]] = []
        self.next_attempt: int = 1
        self.raise_lookup_for: set[str] = set()

    def __call__(self, username: str, score: float) -> int:
        if username in self.raise_lookup_for:
            raise LookupError(
                f"user {username!r} is not in the participants dataset"
            )
        self.calls.append((username, score))
        attempt = self.next_attempt
        self.next_attempt += 1
        return attempt


# ---------------------------------------------------------------------------
# load_test
# ---------------------------------------------------------------------------


class LoadTestShape(unittest.TestCase):

    def test_output_arity_matches_demo_load_outputs(self):
        # 6 fixed outputs + 4 classification-block lists + 3 MCQ-block lists.
        # The output length is a precise invariant the build_demo wiring
        # depends on; a refactor that adds a new state object should bump
        # this together with build_demo's outputs= list.
        out = app.load_test("es", None, participants_df=_PARTS_FRESH)
        expected = (
            6
            + 4 * app.MAX_TEST_QUESTIONS
            + 3 * app.MAX_TEST_MCQ
        )
        self.assertEqual(len(out), expected)

    def test_returns_states_for_questions_and_mcqs(self):
        questions, mcqs, *_ = app.load_test("es", None, participants_df=_PARTS_FRESH)
        # questions_state holds the shuffled classification list…
        self.assertEqual(
            sorted(q["id"] for q in questions),
            sorted(q["id"] for q in load_questions("es")),
        )
        # …mcq_state holds the MCQs in file order.
        self.assertEqual(
            [q["id"] for q in mcqs],
            [q["id"] for q in load_mcq_questions("es")],
        )


class LoadTestAlreadyPassed(unittest.TestCase):

    def test_passed_user_sees_celebration_no_questions(self):
        # Score above threshold → short-circuit. No state, no questions
        # visible, celebratory status_md.
        df = _parts_with_score("alice", 1.0)  # 100% > 75% threshold
        out = app.load_test("es", _prof("alice"), participants_df=df)
        questions_state, mcq_state, intro_upd, status_upd, submit_upd, retake_upd, *_rest = out
        self.assertEqual(questions_state, [])
        self.assertEqual(mcq_state, [])
        # Intro hidden; status carries the "already passed" string.
        self.assertEqual(intro_upd.get("visible"), False)
        self.assertEqual(status_upd.get("value"), app._t("es")["test_already_passed"])
        self.assertEqual(status_upd.get("visible"), True)
        # Submit + retake both hidden — there's nothing to do.
        self.assertEqual(submit_upd.get("visible"), False)
        self.assertEqual(retake_upd.get("visible"), False)


class LoadTestFresh(unittest.TestCase):

    def test_fresh_user_sees_questions(self):
        out = app.load_test("es", _prof("alice"), participants_df=_PARTS_FRESH)
        questions_state, mcq_state, intro_upd, status_upd, submit_upd, retake_upd, *_rest = out
        self.assertEqual(len(questions_state), 14)
        self.assertEqual(len(mcq_state), 2)
        self.assertEqual(intro_upd.get("visible"), True)
        # No previous attempts → no best-score line; status is empty.
        self.assertEqual(status_upd.get("value"), "")
        self.assertEqual(submit_upd.get("visible"), True)
        self.assertEqual(retake_upd.get("visible"), False)

    def test_best_score_line_shown_when_user_has_attempts_but_not_passing(self):
        # 50% = below 75% threshold; the test re-renders but adds the
        # "best score so far" line above the intro.
        df = _parts_with_score("alice", 0.5)
        out = app.load_test("es", _prof("alice"), participants_df=df)
        _, _, _, status_upd, *_ = out
        # 0.5 * 16 = 8; should appear as "8 / 16" somewhere in the line.
        self.assertIn("8 / 16", status_upd.get("value"))


# ---------------------------------------------------------------------------
# submit_test
# ---------------------------------------------------------------------------


def _flat_answers(
    questions: list[dict],
    mcqs: list[dict],
    *,
    classification_value_for: callable,
    mcq_value_for: callable,
) -> tuple:
    """Build the flat ``*answers`` tuple submit_test expects: reject pool
    of length ``MAX_TEST_QUESTIONS``, then accept pool of the same length,
    then MCQ pool of length ``MAX_TEST_MCQ``."""
    rejects = [None] * app.MAX_TEST_QUESTIONS
    accepts = [None] * app.MAX_TEST_QUESTIONS
    mcq_pool = [None] * app.MAX_TEST_MCQ
    for i, q in enumerate(questions):
        value = classification_value_for(q)
        if value is None:
            continue
        # Route to reject- or accept-side pool based on the picked bucket;
        # mirrors what _clear_other_radio enforces in the real UI.
        if value in REJECT_CHOICES:
            rejects[i] = value
        else:
            accepts[i] = value
    for i, q in enumerate(mcqs):
        mcq_pool[i] = mcq_value_for(q)
    return (*rejects, *accepts, *mcq_pool)


class SubmitTestLoggedOut(unittest.TestCase):

    def test_no_profile_returns_login_required(self):
        # No grading happens, no record, no tab updates.
        spy = _RecordSpy()
        with patch("app.record_test_attempt", spy):
            out = app.submit_test([], [], "es", None)
        status_upd, submit_upd, retake_upd, *_ = out
        # Compare against the actual translation rather than hardcoding
        # English — same handler is exercised in every language.
        self.assertEqual(status_upd.get("value"), app._t("es")["test_login_required"])
        self.assertEqual(spy.calls, [])


class SubmitTestEmpty(unittest.TestCase):

    def test_empty_questions_and_mcqs_returns_no_questions_message(self):
        # Defensive branch — load_test returns no_questions only when the
        # language has no bank. submit_test must handle the same shape.
        spy = _RecordSpy()
        with patch("app.record_test_attempt", spy):
            out = app.submit_test([], [], "es", _prof("alice"))
        status_upd, *_ = out
        self.assertEqual(status_upd.get("value"), app._t("es")["test_no_questions"])
        self.assertEqual(spy.calls, [])


class SubmitTestUnanswered(unittest.TestCase):

    def test_any_unanswered_blocks_submission(self):
        questions = load_questions("es")
        mcqs = load_mcq_questions("es")
        # Leave everything blank.
        answers = _flat_answers(
            questions, mcqs,
            classification_value_for=lambda q: None,
            mcq_value_for=lambda q: None,
        )
        spy = _RecordSpy()
        with patch("app.record_test_attempt", spy):
            out = app.submit_test(questions, mcqs, "es", _prof("alice"), *answers)
        status_upd, _submit, _retake, *_tabs = out
        self.assertEqual(status_upd.get("value"), app._t("es")["test_status_unanswered"])
        # Nothing persisted; tabs untouched (noop updates).
        self.assertEqual(spy.calls, [])


class SubmitTestPerfect(unittest.TestCase):

    def test_perfect_passes_records_and_unlocks_tabs(self):
        questions = load_questions("es")
        mcqs = load_mcq_questions("es")
        answers = _flat_answers(
            questions, mcqs,
            classification_value_for=lambda q: q["correct_key"],
            mcq_value_for=lambda q: q["correct"],
        )
        spy = _RecordSpy()
        with patch("app.record_test_attempt", spy):
            out = app.submit_test(questions, mcqs, "es", _prof("alice"), *answers)
        status_upd, submit_upd, retake_upd, tab_w, tab_v, tab_vo = out
        # Recorded once with the perfect fraction.
        self.assertEqual(len(spy.calls), 1)
        username, score = spy.calls[0]
        self.assertEqual(username, "alice")
        self.assertEqual(score, 1.0)
        # Passed message + Submit hidden + Retake shown + all three gated
        # tabs revealed.
        self.assertIn("16 / 16", status_upd.get("value"))
        self.assertEqual(submit_upd.get("visible"), False)
        self.assertEqual(retake_upd.get("visible"), True)
        self.assertEqual(tab_w.get("visible"), True)
        self.assertEqual(tab_v.get("visible"), True)
        self.assertEqual(tab_vo.get("visible"), True)


class SubmitTestFailed(unittest.TestCase):

    def test_below_threshold_does_not_reveal_tabs(self):
        # Pick wrong-side for every classification + wrong MCQ → raw = -9,
        # well below 12. Tabs must NOT be revealed (they get noop updates
        # so a previously-passed user isn't accidentally re-hidden).
        questions = load_questions("es")
        mcqs = load_mcq_questions("es")
        # Wrong side: pick "knowledge" for reject Qs, "trivial" for accept Qs.
        def wrong_side(q):
            return "knowledge" if q["correct_key"] in REJECT_CHOICES else "trivial"

        def wrong_mcq(q):
            return next(opt for opt in q["options"] if opt != q["correct"])

        answers = _flat_answers(
            questions, mcqs,
            classification_value_for=wrong_side,
            mcq_value_for=wrong_mcq,
        )
        spy = _RecordSpy()
        with patch("app.record_test_attempt", spy):
            out = app.submit_test(questions, mcqs, "es", _prof("alice"), *answers)
        status_upd, submit_upd, retake_upd, tab_w, tab_v, tab_vo = out
        # Recorded as a (negative) fraction.
        self.assertEqual(len(spy.calls), 1)
        _user, score = spy.calls[0]
        self.assertLess(score, 0)
        # Failure message contains the needed mark.
        self.assertIn("12 / 16", status_upd.get("value"))
        # Tabs untouched on failure (noop = empty gr.update dict, no
        # ``visible`` key) so any previously-unlocked tab stays unlocked.
        self.assertNotIn("visible", tab_w)
        self.assertNotIn("visible", tab_v)
        self.assertNotIn("visible", tab_vo)


class SubmitTestNotParticipant(unittest.TestCase):

    def test_lookup_error_renders_not_participant_message(self):
        questions = load_questions("es")
        mcqs = load_mcq_questions("es")
        answers = _flat_answers(
            questions, mcqs,
            classification_value_for=lambda q: q["correct_key"],
            mcq_value_for=lambda q: q["correct"],
        )
        spy = _RecordSpy()
        spy.raise_lookup_for = {"stranger"}
        with patch("app.record_test_attempt", spy):
            out = app.submit_test(
                questions, mcqs, "es", _prof("stranger"), *answers
            )
        status_upd, *_ = out
        # Localised "not registered" message, with the username interpolated.
        expected = app._t("es")["test_not_participant"].format(username="stranger")
        self.assertEqual(status_upd.get("value"), expected)


if __name__ == "__main__":
    unittest.main()
