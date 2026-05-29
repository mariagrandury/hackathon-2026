"""CSV / filesystem helpers for the analysis suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plots_dir_for(analyze_py_path: str | Path) -> Path:
    """`plots/` next to an `analyze.py` file. Creates it if missing."""
    return ensure_dir(Path(analyze_py_path).resolve().parent / "plots")


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def section_dir(analyze_py_path: str | Path) -> Path:
    return Path(analyze_py_path).resolve().parent
