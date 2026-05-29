"""Run every analysis in `analysis/`.

Loads the two HF datasets once and passes the DataFrames into each
section's `run(participants, prompts, out_dir)`. Saves the ~4 redundant
Hub fetches the standalone-script form would otherwise do.

Section folders start with a digit (`01_`, `02_`, …), which isn't a
valid Python identifier, so we load each `analyze.py` by file path via
`importlib.util.spec_from_file_location` instead of `import analysis.01_….analyze`.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import data as _data


SECTIONS = [
    ("01_participant_demographics", "demographics"),
    ("02_prompt_writing", "prompt_writing"),
    ("03_prompt_validation", "prompt_validation"),
    ("04_answer_voting", "answer_voting"),
    ("05_entry_test", "entry_test"),
]


def _load_section(folder: str, module_name: str):
    analyze_py = _HERE / folder / "analyze.py"
    spec = importlib.util.spec_from_file_location(
        f"analysis_section_{module_name}", analyze_py
    )
    assert spec and spec.loader, f"could not build spec for {analyze_py}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    t0 = time.monotonic()
    participants = _data.load_participants_df()
    prompts = _data.load_prompts_df()
    print(f"loaded {len(participants)} participants, {len(prompts)} prompts in {time.monotonic() - t0:.1f}s")
    for folder, name in SECTIONS:
        print(f"\n--- {folder} ---")
        section = _load_section(folder, name)
        t = time.monotonic()
        section.run(participants_df=participants, prompts_df=prompts)
        print(f"  ({folder} finished in {time.monotonic() - t:.1f}s)")
    print(f"\nall sections complete in {time.monotonic() - t0:.1f}s total")


if __name__ == "__main__":
    main()
