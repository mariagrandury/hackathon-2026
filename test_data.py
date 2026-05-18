"""Question bank and grading for the entry test.

Questions live in a **private** HF dataset, ``TEST_BANK_REPO``, fetched
at runtime via ``HF_TOKEN``. Keeping the bank out of the Space repo
(where it used to live as ``data/test-2026.json``) prevents anyone
admitted to the private Space from reading the answer key — Space
collaborators can browse all bundled files, even when the Space is
"private". For local dev, the file at ``data/test-2026.json`` is used
as a fallback when the Hub fetch fails (e.g. offline).

The dataset is a single JSON blob at ``test-2026.json`` with the *full*
classification bank, grouped by category in canonical order. The entry
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
import logging
import os
import threading
from collections import defaultdict
from typing import Iterable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from data import HF_TOKEN, REJECT_CHOICES, VALIDATION_CHOICES

log = logging.getLogger("hackathon")

# Private dataset that holds the entry-test bank. Fetched with the same
# ``HF_TOKEN`` that the Space uses for the participants and prompts
# datasets — no separate secret needed.
TEST_BANK_REPO = "mariagrandury/hackathon_test_bank"
# Per-language file inside the dataset. ES is the canonical ``test-2026.json``;
# EN / PT are translations of the prompt framing while keeping
# Spanish-cultural content (idioms, in-character dialogues, region names)
# intact — see ``translate_test_bank.py``. Unknown languages fall back to ES.
TEST_FILES = {
    "es": "test-2026.json",
    "en": "test-2026-en.json",
    "pt": "test-2026-pt.json",
}
_DEFAULT_LANG = "es"
DATA_DIR = "data"  # local fallback only; not synced to the Space
SUPPORTED_FORMAT = "classification"

# Per-language module-level cache of the loaded JSON. The bank changes
# only when an organiser pushes a new version; tab opens shouldn't
# re-download it. The lock serialises first-load attempts (per language)
# under Gradio's threaded server so we don't fire N concurrent
# ``hf_hub_download`` calls for the same file.
_BANK_CACHE: dict[str, dict] = {}
_BANK_LOCK = threading.Lock()

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


def _local_path(lang: str) -> str:
    """Bundled fallback path for dev / offline use."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, DATA_DIR, TEST_FILES.get(lang, TEST_FILES[_DEFAULT_LANG]))


def _resolve_lang(lang: str | None) -> str:
    """Map an arbitrary language string to a key in ``TEST_FILES`` —
    unknown languages get the canonical Spanish bank."""
    if lang and lang in TEST_FILES:
        return lang
    return _DEFAULT_LANG


def _load_bank(lang: str) -> dict:
    """Return the test-bank JSON for ``lang``, fetched from the private HF
    dataset and cached per-language. Falls back to the local bundled file
    (dev / offline) if the Hub fetch fails for any reason.

    Failures with no local fallback return an empty dict WITHOUT caching,
    so a transient token / connectivity issue on Space startup doesn't
    lock the entry test into an empty bank for the rest of the process —
    the next call gets another chance once the token / network recovers."""
    lang = _resolve_lang(lang)
    cached = _BANK_CACHE.get(lang)
    if cached is not None:
        return cached
    with _BANK_LOCK:
        # Re-check under the lock so the second caller of a concurrent
        # first-load doesn't redo the fetch.
        cached = _BANK_CACHE.get(lang)
        if cached is not None:
            return cached
        filename = TEST_FILES[lang]
        try:
            path = hf_hub_download(
                repo_id=TEST_BANK_REPO,
                filename=filename,
                repo_type="dataset",
                token=HF_TOKEN,
            )
            log.info("loaded test bank %s from %s", filename, TEST_BANK_REPO)
        except (HfHubHTTPError, OSError) as exc:
            fallback = _local_path(lang)
            if not os.path.exists(fallback):
                log.warning(
                    "test bank Hub fetch failed (%s) and no local fallback at %s",
                    exc, fallback,
                )
                return {}  # NOT cached — let the next call retry the Hub.
            log.warning("test bank Hub fetch failed (%s); using local %s", exc, fallback)
            path = fallback
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        _BANK_CACHE[lang] = payload
        return payload


def _label_to_key(label: str) -> str | None:
    return _LABEL_TO_KEY.get(label.strip().lower())


def _load_all_classification(lang: str) -> list[dict]:
    """Every classification question in the bank with ``correct_key``
    resolved. Preserves the file's order (which is canonical category
    order). Questions whose ``correct`` label can't be mapped (no
    explicit ``correct_key`` and ``correct`` isn't a known ES label), or
    whose ``format`` isn't ``classification``, are dropped."""
    payload = _load_bank(lang)
    out = []
    for q in payload.get("questions", []):
        if q.get("format") != SUPPORTED_FORMAT:
            continue
        # Prefer the explicit ``correct_key`` written by translate_test_bank.py;
        # fall back to ES label lookup for back-compat with older bank files.
        key = q.get("correct_key") or _label_to_key(q.get("correct", ""))
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


def load_mcq_questions(lang: str) -> list[dict]:
    """Multiple-choice comparison questions from ``multiple_choice[]``.
    Each has ``options`` (4 strings) and ``correct`` (the full text of the
    right one — matches one of ``options``). All of them appear in the
    entry test; there's no per-category quota because they're discriminator
    questions that span categories."""
    return [dict(q) for q in _load_bank(lang).get("multiple_choice", [])]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
#
# Classification questions use a partial-credit scheme (rewards "you got the
# Reject/Accept side right" even when the exact bucket is wrong). MCQs are
# treated as theoretical comprehension questions: binary +1 / -1 with no
# partial credit, since the 4 options aren't side-orderable.

_CLASSIFICATION_EXACT = 1.0
_CLASSIFICATION_SAME_SIDE = 0.5
_CLASSIFICATION_WRONG_SIDE = -0.5
_MCQ_CORRECT = 1.0
_MCQ_WRONG = -1.0


def _score_classification(q: dict, value: str | None) -> float:
    if value is None:
        return 0.0
    correct = q["correct_key"]
    if value == correct:
        return _CLASSIFICATION_EXACT
    same_side = (value in REJECT_CHOICES) == (correct in REJECT_CHOICES)
    return _CLASSIFICATION_SAME_SIDE if same_side else _CLASSIFICATION_WRONG_SIDE


def _score_mcq(q: dict, value: str | None) -> float:
    if value is None:
        return 0.0
    return _MCQ_CORRECT if value == q.get("correct") else _MCQ_WRONG


def grade(
    answers: Iterable[tuple[str, str | None]], lang: str
) -> tuple[float, float, float]:
    """Score the user's answers against the entry test (classification + MCQ).

    ``answers`` is an iterable of ``(question_id, chosen_value)`` pairs.
    For classification questions ``chosen_value`` is a canonical bucket
    key; for MCQs it's the full option string. Unknown ids are skipped.

    Returns ``(score_fraction, raw_score, max_possible)`` where
    ``raw_score`` is the signed sum of per-question scores and
    ``max_possible`` = (# classification + # MCQ). ``score_fraction =
    raw_score / max_possible`` and can be negative (the test penalizes
    wrong-side / wrong-MCQ answers); ``best_test_score`` and the
    pass-mark comparison handle that without special-casing."""
    classification = {q["id"]: q for q in load_questions(lang)}
    mcq = {q["id"]: q for q in load_mcq_questions(lang)}
    max_possible = float(len(classification) + len(mcq))
    if max_possible == 0:
        return 0.0, 0.0, 0.0
    raw = 0.0
    for qid, value in answers:
        if qid in classification:
            raw += _score_classification(classification[qid], value)
        elif qid in mcq:
            raw += _score_mcq(mcq[qid], value)
        # Unknown id: silently skip — could happen if the JSON was edited
        # between the user loading the test and submitting.
    return raw / max_possible, raw, max_possible
