"""Tests for the Claude-annotations layer in ``annotate_claude.py``.

The Claude-only logic lives in ``annotate_claude`` (kept out of ``data.py`` so
human and Claude code stay separate). Covers the pure pending-work predicates
and the upsert/merge logic with the Hub I/O mocked out:

  - pending_validation_ids: skips ids Claude already validated.
  - pending_vote_ids: only ids with both answers, minus ones already voted.
  - load_claude_df: missing repo decays to an empty correctly-shaped frame.
  - write_claude_annotations: creates the repo, validates schema on push,
    and *merges* a later voting pass into the row a validation pass created
    (fields left out of a record don't clobber stored values).
  - region / cultural_topic columns round-trip and merge.
  - VOTE_CHOICES_CLAUDE stays in sync with app.VOTE_CHOICES; CULTURAL_TOPICS
    has an ``other`` catch-all and unique keys.

Run from the repo root::

    python -m unittest tests.test_claude_annotations
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import annotate_claude as ac
import data


def _prompts_df() -> pd.DataFrame:
    """Minimal prompts table — only columns the Claude helpers read."""
    return pd.DataFrame(
        [
            {"id": 1, "answer_a": "x", "answer_b": "y"},   # votable
            {"id": 2, "answer_a": "", "answer_b": ""},     # validation only
            {"id": 3, "answer_a": "x", "answer_b": ""},    # half-answered → not votable
            {"id": 4, "answer_a": "x", "answer_b": "y"},   # votable
        ]
    )


class PendingPredicates(unittest.TestCase):
    def test_pending_validation_skips_done(self):
        claude = pd.DataFrame(
            [{"id": 2, "validation_choice": "knowledge", "vote_choice": ""}],
            columns=list(ac.CLAUDE_FEATURES),
        )
        self.assertEqual(ac.pending_validation_ids(_prompts_df(), claude), [1, 3, 4])

    def test_pending_validation_empty_claude(self):
        self.assertEqual(
            ac.pending_validation_ids(_prompts_df(), ac.empty_claude_df()),
            [1, 2, 3, 4],
        )

    def test_pending_vote_only_answered_minus_done(self):
        claude = pd.DataFrame(
            [{"id": 1, "validation_choice": "", "vote_choice": "a"}],
            columns=list(ac.CLAUDE_FEATURES),
        )
        # ids 1 and 4 are votable; 1 is already voted → only 4 left.
        self.assertEqual(ac.pending_vote_ids(_prompts_df(), claude), [4])

    def test_pending_vote_empty_claude(self):
        self.assertEqual(ac.pending_vote_ids(_prompts_df(), ac.empty_claude_df()), [1, 4])

    def test_nonempty_ids_ignores_blank(self):
        claude = pd.DataFrame(
            [
                {"id": 1, "validation_choice": "  ", "vote_choice": ""},
                {"id": 2, "validation_choice": "trivial", "vote_choice": ""},
            ],
            columns=list(ac.CLAUDE_FEATURES),
        )
        self.assertEqual(ac._nonempty_ids(claude, "validation_choice"), {2})


class LoadClaudeDf(unittest.TestCase):
    def test_missing_repo_returns_empty_shaped(self):
        with patch.object(data, "_fetch_dataset", side_effect=Exception("404")):
            df = ac.load_claude_df()
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), list(ac.CLAUDE_FEATURES))


class WriteAnnotations(unittest.TestCase):
    def _run(self, existing: pd.DataFrame, records: list[dict]):
        """Drive write_claude_annotations with HF I/O mocked, returning the
        DataFrame that would have been pushed."""
        captured = {}

        fake_ds = MagicMock()
        fake_ds.push_to_hub = MagicMock()

        def _from_pandas(df, **kwargs):
            captured["df"] = df.copy()
            return fake_ds

        with patch.object(data, "_hf_api") as api, patch.object(
            ac, "load_claude_df", return_value=existing
        ), patch.object(ac.Dataset, "from_pandas", side_effect=_from_pandas):
            total = ac.write_claude_annotations(records)
            api.return_value.create_repo.assert_called_once()
            # The repo it creates must be the Claude dataset, never a human one.
            self.assertEqual(
                api.return_value.create_repo.call_args.kwargs["repo_id"], ac.CLAUDE_REPO
            )
            fake_ds.push_to_hub.assert_called_once_with(
                ac.CLAUDE_REPO, private=True, token=data.HF_TOKEN
            )
        return total, captured["df"]

    def test_first_write_creates_rows(self):
        total, df = self._run(
            ac.empty_claude_df(),
            [
                {
                    "id": 5,
                    "validation_choice": "preference",
                    "validation_reason": "r",
                    "region": "Cusco",
                    "cultural_topic": "food_and_drink",
                }
            ],
        )
        self.assertEqual(total, 1)
        self.assertEqual(list(df.columns), list(ac.CLAUDE_FEATURES))
        row = df[df["id"] == 5].iloc[0]
        self.assertEqual(row["validation_choice"], "preference")
        self.assertEqual(row["region"], "Cusco")
        self.assertEqual(row["cultural_topic"], "food_and_drink")
        self.assertEqual(row["vote_choice"], "")  # unset stays empty, not NaN

    def test_voting_pass_merges_into_validation_row(self):
        existing = pd.DataFrame(
            [
                {
                    "id": 7,
                    "validation_choice": "dynamics",
                    "validation_reason": "keep me",
                    "vote_choice": "",
                    "vote_reason": "",
                    "region": "Asturias",
                    "cultural_topic": "values_and_opinions",
                    "model": "claude-opus-4-8",
                    "labeled_at": "t0",
                }
            ],
            columns=list(ac.CLAUDE_FEATURES),
        )
        _, df = self._run(existing, [{"id": 7, "vote_choice": "a", "vote_reason": "better"}])
        row = df[df["id"] == 7].iloc[0]
        # Voting pass added the vote WITHOUT clobbering the stored validation,
        # region, or topic.
        self.assertEqual(row["validation_choice"], "dynamics")
        self.assertEqual(row["validation_reason"], "keep me")
        self.assertEqual(row["region"], "Asturias")
        self.assertEqual(row["cultural_topic"], "values_and_opinions")
        self.assertEqual(row["vote_choice"], "a")
        self.assertEqual(row["vote_reason"], "better")

    def test_id_is_int64(self):
        _, df = self._run(ac.empty_claude_df(), [{"id": 9, "validation_choice": "trivial"}])
        self.assertEqual(df["id"].dtype, "int64")


class TaxonomyInvariants(unittest.TestCase):
    def test_topics_unique_and_have_other(self):
        self.assertEqual(len(ac.CULTURAL_TOPICS), len(set(ac.CULTURAL_TOPICS)))
        self.assertIn("other", ac.CULTURAL_TOPICS)

    def test_vote_choices_match_app(self):
        # Neither data nor annotate_claude imports app (Gradio); VOTE_CHOICES_CLAUDE
        # is a copy, so assert it hasn't drifted from the UI's source of truth.
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parent.parent / "app.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "VOTE_CHOICES" for t in node.targets
            ):
                found = tuple(ast.literal_eval(node.value))
                break
        self.assertIsNotNone(found, "VOTE_CHOICES not found in app.py")
        self.assertEqual(found, ac.VOTE_CHOICES_CLAUDE)


if __name__ == "__main__":
    unittest.main()
