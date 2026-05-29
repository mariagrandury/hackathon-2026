"""Generate paper-ready LaTeX `.tex` companions for plots and tables.

Each plot PNG gets a `.tex` next to it with a `\\begin{figure}` env; each
table CSV gets a `.tex` with a `\\begin{table}` env. Captions are written
in Spanish (project language) and carry the contextual information that
USED to live in the plot title — plots themselves render without titles
so the caption is the single source of truth.

Keep the LaTeX surface small: `booktabs` for the tables (toprule /
midrule / bottomrule), plain `\\includegraphics` for the figures. No
TikZ, no exotic packages — drop into a paper that already loads
`graphicx` and `booktabs`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _escape(text: str) -> str:
    """Minimal LaTeX-special-char escaper. Avoids over-escaping (e.g.,
    `\\textbackslash{}` chains) by handling only the common chars."""
    replacements = (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    )
    out = str(text)
    for needle, replacement in replacements:
        out = out.replace(needle, replacement)
    return out


def figure_tex(
    image_filename: str,
    caption: str,
    label: str,
    *,
    width: str = "0.95\\textwidth",
    placement: str = "ht",
) -> str:
    """A `\\begin{figure}` block ready to `\\input` from a paper.

    `image_filename` is the basename of the PNG (no path) — the `.tex`
    lives next to the image so a relative `\\includegraphics{file.png}`
    resolves correctly when the LaTeX `\\input` includes both.
    """
    return (
        f"\\begin{{figure}}[{placement}]\n"
        f"    \\centering\n"
        f"    \\includegraphics[width={width}]{{{image_filename}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{fig:{label}}}\n"
        f"\\end{{figure}}\n"
    )


def _tabular_from_df(
    df: pd.DataFrame,
    *,
    column_format: str | None = None,
    header_labels: Sequence[str] | None = None,
    number_format: str = "{:,.2f}",
) -> str:
    """Render `df` as a `booktabs` tabular env. Numeric cells use
    `number_format` (round to 2 dp by default, with thousands separators
    for ints); strings are LaTeX-escaped."""
    n_cols = df.shape[1]
    fmt = column_format or ("l" + "r" * (n_cols - 1))
    headers = list(header_labels or df.columns)
    head_row = " & ".join(_escape(h) for h in headers) + " \\\\"
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for val in row:
            if pd.isna(val):
                cells.append("--")
            elif isinstance(val, (int,)) and not isinstance(val, bool):
                cells.append(f"{val:,}")
            elif isinstance(val, float):
                if float(val).is_integer():
                    cells.append(f"{int(val):,}")
                else:
                    cells.append(number_format.format(val))
            else:
                cells.append(_escape(val))
        body_rows.append(" & ".join(cells) + " \\\\")
    return (
        f"\\begin{{tabular}}{{{fmt}}}\n"
        "\\toprule\n"
        f"{head_row}\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def table_tex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    column_format: str | None = None,
    header_labels: Sequence[str] | None = None,
    placement: str = "ht",
    number_format: str = "{:,.2f}",
) -> str:
    """A `\\begin{table}` block wrapping a booktabs tabular built from `df`."""
    tabular = _tabular_from_df(
        df,
        column_format=column_format,
        header_labels=header_labels,
        number_format=number_format,
    )
    indented = "    " + tabular.replace("\n", "\n    ").rstrip() + "\n"
    return (
        f"\\begin{{table}}[{placement}]\n"
        f"    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{tab:{label}}}\n"
        f"{indented}"
        f"\\end{{table}}\n"
    )


def save_figure_tex(
    image_path: Path,
    caption: str,
    label: str,
    *,
    width: str = "0.95\\textwidth",
) -> Path:
    """Write a `.tex` next to `image_path` (same basename). Returns the
    path to the new file."""
    image_path = Path(image_path)
    tex_path = image_path.with_suffix(".tex")
    tex_path.write_text(
        figure_tex(image_path.name, caption, label, width=width),
        encoding="utf-8",
    )
    return tex_path


def save_table_tex(
    csv_path: Path,
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    column_format: str | None = None,
    header_labels: Sequence[str] | None = None,
    number_format: str = "{:,.2f}",
) -> Path:
    """Write a `.tex` next to `csv_path` (same basename). The .tex
    embeds a booktabs tabular built from `df` — typically a small
    aggregated SUMMARY of the underlying CSV — so it's paper-ready
    even when the CSV has thousands of rows. The caption should
    reference the CSV by filename so a reader can find the raw data."""
    csv_path = Path(csv_path)
    tex_path = csv_path.with_suffix(".tex")
    tex_path.write_text(
        table_tex(
            df,
            caption,
            label,
            column_format=column_format,
            header_labels=header_labels,
            number_format=number_format,
        ),
        encoding="utf-8",
    )
    return tex_path
