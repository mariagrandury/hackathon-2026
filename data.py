"""Hugging Face dataset I/O for the hackathon Space.

Both the participants and the cultural-preferences datasets are private; the
calling environment must expose ``HF_TOKEN``. On a Hugging Face Space this is
configured as a secret. Locally, drop a ``.env`` file at the repo root with
``HF_TOKEN=hf_...`` and ``python-dotenv`` will load it on import.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from datasets import Dataset, Features, Value, load_dataset
from dotenv import load_dotenv

PARTICIPANTS_REPO = "mariagrandury/hackathon_participants"
PROMPTS_REPO = "mariagrandury/cultural_preferences"

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

EMPTY_VALIDATION = {"choice": "", "username": ""}
EMPTY_VOTE = {"choice": "", "username": ""}

VALIDATION_STRUCT = {"choice": Value("string"), "username": Value("string")}
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
    return all(
        row[f"prompt_validation_{i}"]["choice"] == "relevant"
        for i in (1, 2, 3)
    )


def has_answers(row) -> bool:
    return bool(str(row.get("answer_a", "")).strip()) and bool(
        str(row.get("answer_b", "")).strip()
    )


def user_stats(username: str, df: pd.DataFrame) -> dict:
    """Counts of prompts written, validations recorded, and votes recorded
    by ``username``."""
    if not username or df.empty:
        return {"sent": 0, "validated": 0, "voted": 0}
    sent = int((df["username"] == username).sum())
    validated = int(
        sum(
            df[f"prompt_validation_{i}"]
            .apply(lambda v: v["username"] == username)
            .sum()
            for i in (1, 2, 3)
        )
    )
    voted = int(
        sum(
            df[f"answer_chosen_{i}"]
            .apply(lambda v: v["username"] == username)
            .sum()
            for i in (1, 2, 3)
        )
    )
    return {"sent": sent, "validated": validated, "voted": voted}


def all_known_usernames(df: pd.DataFrame) -> list[str]:
    """Every username that has authored, validated, or voted on a prompt."""
    if df.empty:
        return []
    names: set[str] = set(df["username"].dropna().astype(str))
    for col in (
        "prompt_validation_1",
        "prompt_validation_2",
        "prompt_validation_3",
        "answer_chosen_1",
        "answer_chosen_2",
        "answer_chosen_3",
    ):
        names.update(u for u in df[col].apply(lambda v: v["username"]) if u)
    return sorted(n for n in names if n)


def country_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country: ``fully_validated`` and ``pending`` (sent but not yet
    fully validated). Sums to total prompts sent."""
    if df.empty:
        return pd.DataFrame(columns=["country", "fully_validated", "pending"])
    rows = []
    for country in sorted(df["country"].dropna().unique()):
        sub = df[df["country"] == country]
        fully = int(sub.apply(is_fully_validated, axis=1).sum())
        rows.append(
            {
                "country": country,
                "fully_validated": fully,
                "pending": len(sub) - fully,
            }
        )
    return pd.DataFrame(rows)


def ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per known user with their three counts, sorted by prompts sent."""
    columns = ["username", "prompts sent", "prompts validated", "answers voted"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for username in all_known_usernames(df):
        s = user_stats(username, df)
        rows.append(
            {
                "username": username,
                "prompts sent": s["sent"],
                "prompts validated": s["validated"],
                "answers voted": s["voted"],
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["prompts sent", "prompts validated", "answers voted"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
