"""Numeric helpers for analysis: Wilson CI, unanimity, Lorenz."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. Returns (low, high) clipped
    to [0, 1]. n == 0 returns (0.0, 1.0) so callers can draw a full-width
    error bar instead of a NaN."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def lorenz_points(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """(xs, ys) for a Lorenz curve. xs is cumulative share of population,
    ys is cumulative share of total. Empty input returns the (0,0)-(1,1)
    diagonal."""
    arr = np.sort(np.asarray(list(values), dtype=float))
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.cumsum(arr) / arr.sum()
    xs = np.concatenate([[0.0], np.arange(1, arr.size + 1) / arr.size])
    ys = np.concatenate([[0.0], cum])
    return xs, ys


def gini(values: Iterable[float]) -> float:
    """Gini coefficient in [0, 1]. 0 = perfect equality. Empty / all-zero
    input returns 0.0."""
    arr = np.sort(np.asarray(list(values), dtype=float))
    arr = arr[arr >= 0]
    n = arr.size
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum()))


def unanimity_rate_by_group(
    validations_long: pd.DataFrame,
    *,
    group_col: str = "prompt_language",
    prompt_id_col: str = "prompt_id",
    choice_col: str = "choice",
) -> pd.DataFrame:
    """Per group: among prompts with 3 filled validation slots, % where all
    3 validators picked the same bucket. Returns columns
    `[group_col, n_full, n_unanimous, unanimity_rate]`."""
    if validations_long.empty:
        return pd.DataFrame(columns=[group_col, "n_full", "n_unanimous", "unanimity_rate"])
    counts = validations_long.groupby(prompt_id_col).size()
    full_ids = counts[counts == 3].index
    full = validations_long[validations_long[prompt_id_col].isin(full_ids)].copy()
    if full.empty:
        return pd.DataFrame(columns=[group_col, "n_full", "n_unanimous", "unanimity_rate"])
    per_prompt = (
        full.groupby([prompt_id_col, group_col])[choice_col]
        .nunique()
        .reset_index(name="distinct_choices")
    )
    per_prompt["unanimous"] = per_prompt["distinct_choices"] == 1
    out = (
        per_prompt.groupby(group_col)
        .agg(n_full=("unanimous", "size"), n_unanimous=("unanimous", "sum"))
        .reset_index()
    )
    out["unanimity_rate"] = out["n_unanimous"] / out["n_full"].where(out["n_full"] > 0, 1)
    return out


def accept_reject_agreement(
    validations_long: pd.DataFrame,
    *,
    prompt_id_col: str = "prompt_id",
    side_col: str = "side",
) -> dict[str, float]:
    """Among prompts with >=2 filled validation slots, what fraction had ALL
    validators agree on the accept/reject binary (even if they disagreed on
    the exact bucket)? Returns `{n_prompts, n_agreed, rate}`."""
    if validations_long.empty:
        return {"n_prompts": 0, "n_agreed": 0, "rate": 0.0}
    counts = validations_long.groupby(prompt_id_col).size()
    multi_ids = counts[counts >= 2].index
    sub = validations_long[validations_long[prompt_id_col].isin(multi_ids)]
    if sub.empty:
        return {"n_prompts": 0, "n_agreed": 0, "rate": 0.0}
    sides_per_prompt = sub.groupby(prompt_id_col)[side_col].nunique()
    n_prompts = int(sides_per_prompt.size)
    n_agreed = int((sides_per_prompt == 1).sum())
    rate = n_agreed / n_prompts if n_prompts else 0.0
    return {"n_prompts": n_prompts, "n_agreed": n_agreed, "rate": rate}
