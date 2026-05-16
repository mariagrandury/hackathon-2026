"""Question bank and grading for the entry test.

Questions live in ``data/test-2026.json``. The file holds the *full*
classification bank, grouped by category in canonical order; the entry
test consumes only ``TEST_QUESTIONS_PER_CATEGORY`` per category, and the
rest are reserved as hidden quality-check prompts for validators (see
``load_hidden_questions``)::

    {
      "version": "2026",
      "language": "es",
      "classification_options": ["Trivial / factual", ...],
      "questions": [
        {"id": "T-1", "format": "classification", "prompt": "...",
         "correct": "Trivial / factual"},
        ...   // every classification question, grouped by category
      ],
      "multiple_choice": [...]   // not rendered today
    }

``correct`` is the display label of the right answer; we map it back to
the canonical bucket key (`trivial` / `stereotype` / `unrelated` /
`knowledge` / `preference` / `dynamics` / `bias_probe`) so it can be
compared to the value the Gradio Radio emits.

Non-classification questions (``multiple_choice``) are skipped — they
need more UI than a single radio.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Iterable

from data import VALIDATION_CHOICES

TEST_FILE = "test-2026.json"
DATA_DIR = "data"
SUPPORTED_FORMAT = "classification"

# Per-category quota used by the entry test. The first
# ``TEST_QUESTIONS_PER_CATEGORY`` items in each bucket of ``questions[]``
# are shown to the user; anything past that index is kept for the hidden
# quality-check step (``load_hidden_questions``).
TEST_QUESTIONS_PER_CATEGORY = 2

# Map of display label (as it appears in ``correct``) → canonical bucket
# key. Lower-cased on lookup so small label tweaks don't break grading.
# The keys match ``VALIDATION_CHOICES`` in ``data.py``.
_LABEL_TO_KEY = {
    "trivial / factual": "trivial",
    "reproduce un estereotipo": "stereotype",
    "sin anclaje cultural en el país": "unrelated",
    "conocimiento cultural": "knowledge",
    "preferencia / norma cultural": "preference",
    "dinámica cultural": "dynamics",
    "trampa de sesgo": "bias_probe",
}


def _path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, DATA_DIR, TEST_FILE)


def _label_to_key(label: str) -> str | None:
    return _LABEL_TO_KEY.get(label.strip().lower())


def _load_all_classification(lang: str) -> list[dict]:
    """Every classification question in the file with ``correct_key``
    resolved. Preserves the file's order (which is canonical category
    order). Questions whose ``correct`` label can't be mapped, or whose
    ``format`` isn't ``classification``, are dropped.

    Only Spanish ships today; ``lang`` is reserved for the upcoming
    per-language banks (see the Entry-test follow-ups in CLAUDE.md)."""
    path = _path()
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    out = []
    for q in payload.get("questions", []):
        if q.get("format") != SUPPORTED_FORMAT:
            continue
        key = _label_to_key(q.get("correct", ""))
        if key is None:
            continue
        out.append({**q, "correct_key": key})
    return out


def _grouped_by_category(questions: list[dict]) -> dict[str, list[dict]]:
    """Group ``questions`` by ``correct_key``, preserving in-bucket order."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        grouped[q["correct_key"]].append(q)
    return grouped


def load_questions(lang: str) -> list[dict]:
    """Entry-test bank: the first ``TEST_QUESTIONS_PER_CATEGORY`` questions
    of each bucket, in ``VALIDATION_CHOICES`` order. Today that's 1 × 7 = 7
    questions. The remaining classification items stay in the file and are
    returned by ``load_hidden_questions``."""
    grouped = _grouped_by_category(_load_all_classification(lang))
    out: list[dict] = []
    for key in VALIDATION_CHOICES:
        out.extend(grouped.get(key, [])[:TEST_QUESTIONS_PER_CATEGORY])
    return out


def load_hidden_questions(lang: str) -> list[dict]:
    """Questions reserved for the hidden quality-check step (next milestone):
    everything in each bucket *past* ``TEST_QUESTIONS_PER_CATEGORY``. These
    will be inserted as the first prompts each validator sees, so we can
    spot validators who annotate at random."""
    grouped = _grouped_by_category(_load_all_classification(lang))
    out: list[dict] = []
    for key in VALIDATION_CHOICES:
        out.extend(grouped.get(key, [])[TEST_QUESTIONS_PER_CATEGORY:])
    return out


def grade(answers: Iterable[tuple[str, str | None]], lang: str) -> tuple[float, int, int]:
    """Score the user's answers against the entry-test bank.

    ``answers`` is an iterable of ``(question_id, chosen_value)`` pairs,
    where ``chosen_value`` is a canonical bucket key. Returns
    ``(score_fraction, n_correct, n_total)``; unanswered questions count
    against the score. The denominator is the entry-test size (not the
    full bank) — hidden questions don't affect the score."""
    questions = {q["id"]: q for q in load_questions(lang)}
    total = len(questions)
    if total == 0:
        return 0.0, 0, 0
    correct = 0
    for qid, value in answers:
        q = questions.get(qid)
        if q is None:
            continue
        if value is not None and value == q["correct_key"]:
            correct += 1
    return correct / total, correct, total
