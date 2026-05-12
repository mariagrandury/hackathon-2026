"""Migrate prompts from the local 2025 dataset into ``cultural_preferences``.

Source: ``data/dataset-2025.json`` — 2136 rows grouped by country (full
Spanish names at the top level). Each row has ``system_prompt``,
``user_message``, ``accepted_response``/``rejected_response`` with their
corresponding ``accepted_model`` / ``rejected_model``, plus the original
annotator's ``username``, ``tie`` and ``both_bad`` flags.

Per-row mapping to our schema:
  * ``username``     → ``"v0"`` (sentinel — the prompts don't have a
    real author; using ``"v0"`` keeps the "can't validate your own
    prompt" rule out of the way for everyone).
  * ``language``     → ``"es"`` (every country in the dataset is
    Spanish-speaking).
  * ``country``      → 2-letter ISO from ``COUNTRY_MAP``.
  * ``system_prompt`` → ``src.system_prompt`` (shown to annotators
    merged with the prompt in the validation + voting tabs).
  * ``prompt``       → ``src.user_message``.
  * ``answer_a`` / ``answer_b`` → if ``tie`` or ``both_bad`` is True,
    *both* are left empty (prompt still goes in, but the voting tab
    won't offer it until someone supplies a new pair of answers).
    Otherwise ``accepted_response`` / ``rejected_response`` are placed
    in random order per row (seeded), so voters don't see a fixed A/B
    correlation with the original annotator's pick.
  * ``model_a`` / ``model_b`` → matching ``accepted_model`` /
    ``rejected_model`` after the swap, or both empty for tie/both_bad.
  * validation + vote slots → all empty (existing flow applies: 3
    ``relevant`` validations before the voting tab will offer the prompt).

Fields *not* propagated:
  * The original annotator's ``username`` (an email) — irrelevant; the
    prompts have no real author in our model.
  * ``tie`` / ``both_bad`` themselves — encoded in the absence of
    answers; the hackathon participants will re-judge from scratch.

DESTRUCTIVE: overwrites ``mariagrandury/cultural_preferences`` wholesale.
Re-running drops any user annotations already collected on that repo.

Usage:
    python migrate_from_v0.py          # uses HF_TOKEN from env / .env
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from data import (
    EMPTY_VALIDATION,
    EMPTY_VOTE,
    PROMPTS_FEATURES,
    PROMPTS_REPO,
)

LOCAL_DATASET_PATH = Path(__file__).resolve().parent / "data" / "dataset-2025.json"
SEED = 20260512

# Spanish-name → 2-letter ISO. Listed in order of frequency in the dataset.
COUNTRY_MAP = {
    "España": "es",
    "Cuba": "cu",
    "Colombia": "co",
    "Paraguay": "py",
    "Ecuador": "ec",
    "Chile": "cl",
    "Perú": "pe",
    "México": "mx",
    "Nicaragua": "ni",
}


def _row(src_row: dict, country_iso: str, rng: random.Random) -> dict:
    # tie/both_bad rows still get migrated (the prompt is useful) but the
    # answer pair is dropped — the voting tab gates on has_answers(), so a
    # row with empty answers will sit in validation only until someone
    # supplies a fresh pair out-of-band.
    drop_answers = bool(src_row.get("tie")) or bool(src_row.get("both_bad"))

    if drop_answers:
        answer_a = answer_b = ""
        model_a = model_b = ""
    else:
        accepted_is_a = rng.random() < 0.5
        if accepted_is_a:
            answer_a = src_row["accepted_response"]
            model_a = src_row["accepted_model"]
            answer_b = src_row["rejected_response"]
            model_b = src_row["rejected_model"]
        else:
            answer_a = src_row["rejected_response"]
            model_a = src_row["rejected_model"]
            answer_b = src_row["accepted_response"]
            model_b = src_row["accepted_model"]

    return {
        "username": "v0",
        "language": "es",
        "country": country_iso,
        "system_prompt": src_row.get("system_prompt", ""),
        "prompt": src_row["user_message"],
        "prompt_validation_1": dict(EMPTY_VALIDATION),
        "prompt_validation_2": dict(EMPTY_VALIDATION),
        "prompt_validation_3": dict(EMPTY_VALIDATION),
        "answer_a": answer_a,
        "model_a": model_a,
        "answer_b": answer_b,
        "model_b": model_b,
        "answer_chosen_1": dict(EMPTY_VOTE),
        "answer_chosen_2": dict(EMPTY_VOTE),
        "answer_chosen_3": dict(EMPTY_VOTE),
    }


def main() -> None:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Put it in .env or export it before running."
        )

    print(f"Reading {LOCAL_DATASET_PATH}…")
    with LOCAL_DATASET_PATH.open(encoding="utf-8") as f:
        by_country = json.load(f)
    total = sum(len(v) for v in by_country.values())
    print(f"  got {total} rows across {len(by_country)} countries")

    missing = set(by_country) - set(COUNTRY_MAP)
    if missing:
        raise SystemExit(
            f"Countries with no mapping in COUNTRY_MAP: {sorted(missing)}. "
            f"Add them and re-run."
        )

    rng = random.Random(SEED)
    rows = []
    for country_name, country_rows in by_country.items():
        country_iso = COUNTRY_MAP[country_name]
        for r in country_rows:
            rows.append(_row(r, country_iso, rng))

    print(f"Pushing {len(rows)} rows to {PROMPTS_REPO} (overwriting)…")
    ds = Dataset.from_list(rows, features=PROMPTS_FEATURES)
    ds.push_to_hub(PROMPTS_REPO, private=True, token=token)
    print("Done.")


if __name__ == "__main__":
    main()
