"""Hugging Face dataset I/O for the hackathon Space.

Both the participants and the cultural-preferences datasets are private; the
calling environment must expose ``HF_TOKEN`` (Space secret or local export).
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from datasets import Dataset, Features, Value, load_dataset

PARTICIPANTS_REPO = "mariagrandury/hackathon_participants"
PROMPTS_REPO = "mariagrandury/cultural_preferences"

HF_TOKEN = os.environ.get("HF_TOKEN")

EMPTY_VALIDATION = {"validated": False, "username": ""}
EMPTY_VOTE = {"choice": "", "username": ""}

VALIDATION_STRUCT = {"validated": Value("bool"), "username": Value("string")}
VOTE_STRUCT = {"choice": Value("string"), "username": Value("string")}

PARTICIPANTS_FEATURES = Features(
    {
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "gmail": Value("string"),
    }
)

PROMPTS_FEATURES = Features(
    {
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "prompt": Value("string"),
        "prompt_validation_1": VALIDATION_STRUCT,
        "prompt_validation_2": VALIDATION_STRUCT,
        "prompt_validation_3": VALIDATION_STRUCT,
        "answer_a": Value("string"),
        "model_a": Value("string"),
        "answer_b": Value("string"),
        "model_b": Value("string"),
        "answer_chosen_1": VOTE_STRUCT,
        "answer_chosen_2": VOTE_STRUCT,
        "answer_chosen_3": VOTE_STRUCT,
    }
)


def load_participants_df() -> pd.DataFrame:
    return load_dataset(
        PARTICIPANTS_REPO, split="train", token=HF_TOKEN
    ).to_pandas()


def load_prompts_df() -> pd.DataFrame:
    return load_dataset(PROMPTS_REPO, split="train", token=HF_TOKEN).to_pandas()


def push_prompts_df(df: pd.DataFrame) -> None:
    Dataset.from_pandas(
        df, preserve_index=False, features=PROMPTS_FEATURES
    ).push_to_hub(PROMPTS_REPO, private=True, token=HF_TOKEN)


def participant_info(username: str) -> Optional[dict]:
    df = load_participants_df()
    matches = df[df["username"] == username]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def is_fully_validated(row) -> bool:
    return all(row[f"prompt_validation_{i}"]["validated"] for i in (1, 2, 3))


def has_answers(row) -> bool:
    return bool(str(row.get("answer_a", "")).strip()) and bool(
        str(row.get("answer_b", "")).strip()
    )
