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
        # Stable, human-readable identifier (1-indexed, sequential). Used in
        # the commit message of every save so the HF dataset's commit
        # history doubles as an audit log.
        "id": Value("int64"),
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        # Optional LLM-steering preamble shown to annotators alongside the
        # prompt. Empty string when the prompt is self-contained.
        "system_prompt": Value("string"),
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

# 2-letter ISO → display name for the country shown to annotators (in the
# in-progress status messages). Falls back to the uppercase code if a
# country isn't in the map.
COUNTRY_DISPLAY_NAMES: dict[str, str] = {
    "es": "España",
    "cu": "Cuba",
    "co": "Colombia",
    "py": "Paraguay",
    "ec": "Ecuador",
    "cl": "Chile",
    "pe": "Perú",
    "mx": "México",
    "ni": "Nicaragua",
    "br": "Brasil",
    "pt": "Portugal",
    "us": "USA",
    "ar": "Argentina",
    "uy": "Uruguay",
    "ve": "Venezuela",
    "bo": "Bolivia",
    "cr": "Costa Rica",
    "do": "República Dominicana",
    "gt": "Guatemala",
    "hn": "Honduras",
    "pa": "Panamá",
    "pr": "Puerto Rico",
    "sv": "El Salvador",
}


def country_display(code: str | None) -> str:
    if not code:
        return "?"
    return COUNTRY_DISPLAY_NAMES.get(code, code.upper())


def load_participants_df() -> pd.DataFrame:
    return load_dataset(PARTICIPANTS_REPO, split="train", token=HF_TOKEN).to_pandas()


def load_prompts_df() -> pd.DataFrame:
    return load_dataset(PROMPTS_REPO, split="train", token=HF_TOKEN).to_pandas()


def push_prompts_df(
    df: pd.DataFrame, commit_message: str | None = None
) -> None:
    """Push ``df`` to the prompts repo with an optional commit message.

    Callers pass a per-action message like
    ``"mariagrandury validated prompt with ID 42"`` so the HF repo's commit
    history reads like an activity log (visible on huggingface.co)."""
    Dataset.from_pandas(
        df, preserve_index=False, features=PROMPTS_FEATURES
    ).push_to_hub(
        PROMPTS_REPO,
        private=True,
        token=HF_TOKEN,
        commit_message=commit_message,
    )


def participant_info(username: str) -> Optional[dict]:
    df = load_participants_df()
    matches = df[df["username"] == username]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def is_fully_validated(row) -> bool:
    return all(row[f"prompt_validation_{i}"]["choice"] == "relevant" for i in (1, 2, 3))


def has_answers(row) -> bool:
    return bool(str(row.get("answer_a", "")).strip()) and bool(
        str(row.get("answer_b", "")).strip()
    )


VALIDATION_COLS = (
    "prompt_validation_1",
    "prompt_validation_2",
    "prompt_validation_3",
)
VOTE_COLS = ("answer_chosen_1", "answer_chosen_2", "answer_chosen_3")


def _validator_usernames(df: pd.DataFrame) -> pd.Series:
    parts = [df[col].str["username"] for col in VALIDATION_COLS]
    s = pd.concat(parts, ignore_index=True)
    return s[s != ""]


def _voter_usernames(df: pd.DataFrame) -> pd.Series:
    parts = [df[col].str["username"] for col in VOTE_COLS]
    s = pd.concat(parts, ignore_index=True)
    return s[s != ""]


def _fully_validated_mask(df: pd.DataFrame) -> pd.Series:
    """Vectorised equivalent of ``df.apply(is_fully_validated, axis=1)``."""
    return (
        (df["prompt_validation_1"].str["choice"] == "relevant")
        & (df["prompt_validation_2"].str["choice"] == "relevant")
        & (df["prompt_validation_3"].str["choice"] == "relevant")
    )


def user_stats(username: str, df: pd.DataFrame) -> dict:
    """Counts of prompts written, validations recorded, and votes recorded
    by ``username``."""
    if not username or df.empty:
        return {"sent": 0, "validated": 0, "voted": 0}
    sent = int((df["username"] == username).sum())
    validated = int(
        sum((df[col].str["username"] == username).sum() for col in VALIDATION_COLS)
    )
    voted = int(sum((df[col].str["username"] == username).sum() for col in VOTE_COLS))
    return {"sent": sent, "validated": validated, "voted": voted}


def all_known_usernames(df: pd.DataFrame) -> list[str]:
    """Every username that has authored, validated, or voted on a prompt."""
    if df.empty:
        return []
    names: set[str] = set(df["username"].dropna().astype(str))
    names.update(_validator_usernames(df))
    names.update(_voter_usernames(df))
    return sorted(n for n in names if n)


def country_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-country: ``fully_validated`` and ``pending`` (sent but not yet
    fully validated). Sums to total prompts sent."""
    if df.empty:
        return pd.DataFrame(columns=["country", "fully_validated", "pending"])
    grouped = (
        df.assign(_fully=_fully_validated_mask(df))
        .dropna(subset=["country"])
        .groupby("country", sort=True)
        .agg(total=("country", "size"), fully_validated=("_fully", "sum"))
        .reset_index()
    )
    grouped["pending"] = grouped["total"] - grouped["fully_validated"]
    return grouped[["country", "fully_validated", "pending"]].astype(
        {"fully_validated": int, "pending": int}
    )


def ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per known user with their three counts, sorted by prompts sent."""
    columns = ["username", "prompts sent", "prompts validated", "answers voted"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    sent = df["username"].value_counts()
    validated = _validator_usernames(df).value_counts()
    voted = _voter_usernames(df).value_counts()
    users = sorted(set(sent.index) | set(validated.index) | set(voted.index))
    out = pd.DataFrame(
        {
            "username": users,
            "prompts sent": [int(sent.get(u, 0)) for u in users],
            "prompts validated": [int(validated.get(u, 0)) for u in users],
            "answers voted": [int(voted.get(u, 0)) for u in users],
        },
        columns=columns,
    )
    return out.sort_values(
        ["prompts sent", "prompts validated", "answers voted"],
        ascending=False,
    ).reset_index(drop=True)
