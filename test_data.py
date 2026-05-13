"""Question bank and grading for the entry test.

Questions live in ``data/test-2026.json`` and look like::

    {
      "version": "2026",
      "language": "es",
      "classification_options": ["Trivial / factual", ...],
      "questions": [
        {"id": "Q1", "format": "classification", "prompt": "...",
         "correct": "Conocimiento cultural"},
        ...
      ]
    }

``correct`` is the display label of the right answer; we map it back to
the canonical bucket key (`trivial` / `stereotype` / `unrelated` /
`knowledge` / `preference` / `dynamics` / `bias_probe`) so it can be
compared to the value the Gradio Radio emits.

Non-classification questions (e.g. ``multiple_choice``) are skipped — they
need more UI than a single radio.
"""

from __future__ import annotations

import json
import os
from typing import Iterable

TEST_FILE = "test-2026.json"
DATA_DIR = "data"
SUPPORTED_FORMAT = "classification"

# Map of display label (as it appears in ``correct``) → canonical bucket
# key. Lower-cased on lookup so small label tweaks don't break grading.
# The keys match ``VALIDATION_CHOICES`` in ``app.py``.
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


def load_questions(lang: str) -> list[dict]:
    """Load the question list.

    Only Spanish ships today — other languages reuse the same file so the
    UI still works while translations are pending. Each returned question
    has the original fields plus ``correct_key`` (canonical bucket).
    Questions whose ``correct`` label can't be mapped, or whose ``format``
    we don't render, are dropped.
    """
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


def grade(answers: Iterable[tuple[str, str | None]], lang: str) -> tuple[float, int, int]:
    """Score the user's answers against the question bank.

    ``answers`` is an iterable of ``(question_id, chosen_value)`` pairs,
    where ``chosen_value`` is a canonical bucket key. Returns
    ``(score_fraction, n_correct, n_total)``; unanswered questions count
    against the score.
    """
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
