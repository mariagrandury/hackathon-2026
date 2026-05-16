"""Tests for ``data.record_test_attempt``'s commit-style update logic.

Patches out the four Hub-touching points (``HfApi``,
``_participants_parquet_path``, ``load_dataset``, ``_commit_participants``)
so we can exercise the retry/error paths without network or HF state:

  - happy path: fresh user → attempt 1, commit called once with the
    right ``parent_commit`` SHA.
  - increment: user with previous attempts → attempt = max(existing)+1,
    and the JSON in the new row carries every prior score plus the new one.
  - LookupError: user not in the participants df → raised before any commit.
  - 412 retry: first commit conflicts (412), second succeeds → attempt
    still 1 (no double-record), at least two ``dataset_info`` reads.
  - non-412 propagates: other HfHubHTTPError status codes re-raise.
  - all retries fail → ``RuntimeError`` after ``_COMMIT_MAX_RETRIES``.

Run from the repo root::

    python -m unittest tests.test_record_test_attempt
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

import data
from data import _COMMIT_MAX_RETRIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FAKE_SHA = "abc123"
_FAKE_PARQUET_PATH = "data/train-00000-of-00001.parquet"


def _participants_df(scores_by_user: dict[str, str]) -> pd.DataFrame:
    """A minimal participants table — only the columns
    ``record_test_attempt`` actually reads or writes."""
    return pd.DataFrame(
        [
            {"username": u, "language": "es", "country": "es",
             "gmail": "", "test_score": scores}
            for u, scores in scores_by_user.items()
        ]
    )


def _http_error(status: int) -> data.HfHubHTTPError:
    """Build an ``HfHubHTTPError`` whose ``.response.status_code`` matches
    what the retry path inspects. The constructor signature varies across
    versions; assign ``response`` after construction to stay compatible."""
    exc = data.HfHubHTTPError(f"{status} fake error")
    exc.response = SimpleNamespace(status_code=status)
    return exc


@contextmanager
def _patch_io(*, df: pd.DataFrame, commit_side_effect=None):
    """Patch every Hub-touching point in ``data`` for the duration of the
    ``with`` block and yield a ``SimpleNamespace`` of useful handles:

      - ``api``: the ``HfApi`` instance ``record_test_attempt`` constructs.
      - ``commit``: the mocked ``_commit_participants`` (assert call_count
        / call_args here).

    ``commit_side_effect`` is forwarded to the mocked ``_commit_participants``
    (``None`` → succeeds; ``Exception`` → raised; list → raised one per
    successive call)."""
    api = MagicMock()
    api.dataset_info.return_value = SimpleNamespace(sha=_FAKE_SHA)
    fake_ds = SimpleNamespace(to_pandas=lambda: df.copy())
    with (
        patch("data.HfApi", return_value=api),
        patch("data._participants_parquet_path", return_value=_FAKE_PARQUET_PATH),
        patch("data.load_dataset", return_value=fake_ds),
        patch("data._commit_participants", side_effect=commit_side_effect) as commit,
        # Disable backoff sleeps so the "all retries fail" test doesn't
        # actually wait 13+ seconds for the exponential schedule.
        patch("data.time.sleep"),
    ):
        yield SimpleNamespace(api=api, commit=commit)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class HappyPath(unittest.TestCase):

    def test_fresh_user_returns_attempt_1(self):
        df = _participants_df({"alice": "{}"})
        with _patch_io(df=df) as patches:
            attempt = data.record_test_attempt("alice", 0.95)
        self.assertEqual(attempt, 1)
        commit = patches.commit
        commit.assert_called_once()
        # Commit went out with parent_commit = the SHA we read.
        self.assertEqual(commit.call_args.kwargs["parent_commit"], _FAKE_SHA)
        # Path is whatever _participants_parquet_path resolved.
        self.assertEqual(commit.call_args.kwargs["path_in_repo"], _FAKE_PARQUET_PATH)

    def test_commit_message_includes_username_attempt_and_score(self):
        df = _participants_df({"alice": "{}"})
        with _patch_io(df=df) as patches:
            data.record_test_attempt("alice", 0.95)
        msg = patches.commit.call_args.kwargs["commit_message"]
        self.assertIn("alice", msg)
        self.assertIn("attempt 1", msg)
        self.assertIn("0.95", msg)

    def test_new_score_persisted_into_test_score_cell(self):
        df = _participants_df({"alice": "{}"})
        with _patch_io(df=df) as patches:
            data.record_test_attempt("alice", 0.42)
        # The mutated df is the first positional arg to _commit_participants.
        commit = patches.commit
        committed_df = commit.call_args.args[0]
        scores = json.loads(committed_df.loc[committed_df["username"] == "alice", "test_score"].iloc[0])
        self.assertEqual(scores, {"1": 0.42})

    def test_other_users_test_score_left_alone(self):
        df = _participants_df({"alice": "{}", "bob": '{"1": 0.5, "2": 0.7}'})
        with _patch_io(df=df) as patches:
            data.record_test_attempt("alice", 1.0)
        committed_df = patches.commit.call_args.args[0]
        bob_row = committed_df.loc[committed_df["username"] == "bob", "test_score"].iloc[0]
        # Bob's untouched JSON should round-trip exactly.
        self.assertEqual(json.loads(bob_row), {"1": 0.5, "2": 0.7})


class AttemptIncrement(unittest.TestCase):

    def test_existing_attempts_increment(self):
        # max existing key is 2 → new attempt = 3.
        df = _participants_df({"alice": '{"1": 0.5, "2": 0.7}'})
        with _patch_io(df=df) as patches:
            attempt = data.record_test_attempt("alice", 0.9)
        self.assertEqual(attempt, 3)
        committed_df = patches.commit.call_args.args[0]
        scores = json.loads(committed_df.loc[committed_df["username"] == "alice", "test_score"].iloc[0])
        self.assertEqual(scores, {"1": 0.5, "2": 0.7, "3": 0.9})

    def test_non_sequential_attempt_keys_still_increment_from_max(self):
        # Defensive: if attempts get sparse for any reason ("3" present but
        # "1" / "2" missing), next attempt should still be max+1 = 4.
        df = _participants_df({"alice": '{"3": 0.6}'})
        with _patch_io(df=df):
            attempt = data.record_test_attempt("alice", 0.7)
        self.assertEqual(attempt, 4)


# ---------------------------------------------------------------------------
# LookupError
# ---------------------------------------------------------------------------


class MissingUser(unittest.TestCase):

    def test_unknown_user_raises_lookup_error(self):
        df = _participants_df({"alice": "{}"})
        with _patch_io(df=df) as patches:
            with self.assertRaises(LookupError):
                data.record_test_attempt("nobody", 1.0)
        # No commit attempted when the user isn't registered.
        patches.commit.assert_not_called()


# ---------------------------------------------------------------------------
# Retry on 412
# ---------------------------------------------------------------------------


class RetryOn412(unittest.TestCase):

    def test_one_conflict_then_success(self):
        # First commit raises 412 (another writer landed in between); the
        # retry succeeds on the second attempt. Returned attempt number is
        # still 1 because the *participant's* attempt counter advances by
        # the bank state, not by retry count.
        df = _participants_df({"alice": "{}"})
        side_effects = [_http_error(412), None]  # raise, then succeed
        with _patch_io(df=df, commit_side_effect=side_effects) as patches:
            attempt = data.record_test_attempt("alice", 1.0)
        self.assertEqual(attempt, 1)
        # Two commits attempted; two SHA reads.
        self.assertEqual(patches.commit.call_count, 2)
        self.assertGreaterEqual(patches.api.dataset_info.call_count, 2)

    def test_multiple_conflicts_then_success(self):
        # 3 conflicts in a row, then succeed. Confirms we're not capping
        # too early (e.g. via the wrong upper bound in the loop).
        df = _participants_df({"alice": "{}"})
        side_effects = [_http_error(412), _http_error(412), _http_error(412), None]
        with _patch_io(df=df, commit_side_effect=side_effects) as patches:
            attempt = data.record_test_attempt("alice", 0.5)
        self.assertEqual(attempt, 1)
        self.assertEqual(patches.commit.call_count, 4)


class NonRetryableError(unittest.TestCase):

    def test_500_propagates_immediately(self):
        # Anything that isn't a 412 should re-raise; we don't want to mask
        # auth errors / quota errors / etc. behind retries.
        df = _participants_df({"alice": "{}"})
        with _patch_io(df=df, commit_side_effect=_http_error(500)) as patches:
            with self.assertRaises(data.HfHubHTTPError) as ctx:
                data.record_test_attempt("alice", 0.9)
        self.assertEqual(ctx.exception.response.status_code, 500)
        # Single attempt — no retries for non-412.
        self.assertEqual(patches.commit.call_count, 1)


class AllRetriesExhausted(unittest.TestCase):

    def test_runtime_error_after_max_retries(self):
        # Every commit hits 412 → after _COMMIT_MAX_RETRIES, give up with
        # a RuntimeError chained from the last conflict.
        df = _participants_df({"alice": "{}"})
        side_effects = [_http_error(412)] * (_COMMIT_MAX_RETRIES + 2)
        with _patch_io(df=df, commit_side_effect=side_effects) as patches:
            with self.assertRaises(RuntimeError) as ctx:
                data.record_test_attempt("alice", 0.9)
        # Chained from an HfHubHTTPError (the last 412).
        self.assertIsInstance(ctx.exception.__cause__, data.HfHubHTTPError)
        # Hit the loop cap exactly — not infinite, not short-circuited.
        self.assertEqual(
            patches.commit.call_count, _COMMIT_MAX_RETRIES
        )


if __name__ == "__main__":
    unittest.main()
