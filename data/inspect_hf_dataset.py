"""Download both hackathon datasets and plot an overview of each.

Writes two PNGs into ``data/``:

  * ``participants_overview.png`` — language and country from the
    ``hackathon_participants`` dataset, plus a pie chart per demographic
    question (pronouns, education, field, year of birth, NLP level, first
    SomosNLP event) joined in from the Eventbrite registration CSV by HF
    username (case-insensitive — email is no longer pushed to the Hub).
  * ``prompts_overview.png`` — the same views as the app's Leaderboard tab:
    overall totals, the per-user ranking (top 10, stacked), and per-country
    validation status, plus the unique-annotator count.

Demographics need an Eventbrite report CSV; by default the newest
``reports/report-*.csv`` is used (override with ``--eventbrite-csv``). If
none is found, only language and country are plotted.

Needs ``HF_TOKEN`` with read access to the two private datasets — ``data.py``
loads it from ``.env``. Run from the repo root:

    python inspect_hf_dataset.py
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# Repo root is one level up (this script lives in ``data/``). Add it to
# sys.path so ``import data`` resolves to the data.py module, not this
# directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import (
    country_counts,
    country_display,
    load_participants_df,
    load_prompts_df,
    ranking_df,
)
from import_participants_info import _order_by_lang, clean_username

REPO_DIR = _REPO_ROOT
OUT_DIR = REPO_DIR / "data"

# Mirror the app's Leaderboard palette (app._user_progress_color_map /
# app._country_color_map) so this script and the live tab read alike.
C_SENT, C_VALIDATED, C_VOTED = "#3b82f6", "#10b981", "#f59e0b"
C_FULLY, C_PENDING = "#22c55e", "#facc15"


# --- demographic normalizers ------------------------------------------------
# The Eventbrite form asks each question in ES / PT / EN; the free-text answers
# come back in whatever language the registrant used. Each normalizer collapses
# the variants of one question down to a small set of canonical buckets so the
# pie charts stay readable.


def norm_pronouns(v: str) -> str:
    v = v.lower()
    if v in ("él", "ele") or "he/him" in v:
        return "he / él"
    if v == "ella" or "she/her" in v:
        return "she / ella"
    if "no decir" in v or "prefer not" in v:
        return "prefer not to say"
    return v[:20]


def norm_education(v: str) -> str:
    v = v.lower()
    if "doctor" in v:
        return "Tertiary — Doctoral"
    if "máster" in v or "maestr" in v or "magíster" in v or "master" in v:
        return "Tertiary — Master's"
    if "grado" in v or "licenc" in v or "bachelor" in v or "graduação" in v or "bacharel" in v:
        return "Tertiary — Bachelor's"
    if "ciclo corto" in v or "técnica" in v or "short-cycle" in v:
        return "Tertiary — short-cycle"
    if "post-secondary" in v or "non-tertiary" in v:
        return "Post-secondary non-tertiary"
    # checked before upper secondary: "junior high school" contains "high school"
    if "secundaria baja" in v or "middle school" in v or "junior high" in v:
        return "Lower secondary"
    if "secundaria alta" in v or "preparatoria" in v or "bachiller" in v or "high school" in v:
        return "Upper secondary"
    return v[:28] or "(no answer)"


def norm_field(v: str) -> str:
    v = v.lower()
    if "informática" in v or "computer science" in v or "computação" in v:
        return "Computer science"
    if "ingenier" in v or "engineering" in v:
        return "Engineering"
    if "naturales" in v or "natural sciences" in v:
        return "Natural sci. & maths"
    if "artes" in v or "humanidades" in v or "arts" in v or "humanities" in v:
        return "Arts & humanities"
    if "sociales" in v or "periodismo" in v or "social sciences" in v or "journalism" in v:
        return "Social sci. & journalism"
    if "educación" in v or "educação" in v or "education" in v:
        return "Education"
    if "administración" in v or "empresariales" in v or "derecho" in v or "business" in v or "law" in v:
        return "Business & law"
    if "servicios" in v or "services" in v:
        return "Services"
    if "agricultura" in v or "forestal" in v or "agriculture" in v or "veterinar" in v:
        return "Agriculture & veterinary"
    if "salud" in v or "health" in v or "bienestar" in v or "welfare" in v:
        return "Health & welfare"
    if "otro" in v or "other" in v:
        return "Other"
    return v[:24]  # unrecognised value — visible signal that a bucket is missing


def norm_nlp(v: str) -> str:
    v = v.lower()
    if "básico" in v or "basic" in v:
        return "Basic"
    if "intermedi" in v:
        return "Intermediate"
    if "avanzado" in v or "advanced" in v:
        return "Advanced"
    return v[:20]


def norm_first_time(v: str) -> str:
    v = v.lower()
    if v.startswith("s") or "primer" in v:
        return "First SomosNLP event"
    if v.startswith("n") or "anteriores" in v or "asistido" in v:
        return "Returning attendee"
    return v[:24]


def year_bucket(v: str) -> str:
    """Free-text birth years are messy (``2003.0``, ``19/05/2000``, …); pull
    the first 4-digit year out and bucket it."""
    m = re.search(r"(?:19|20)\d{2}", str(v))
    if not m:
        return "(no answer)"
    y = int(m.group(0))
    if y < 1990:
        return "before 1990"
    if y < 1995:
        return "1990–1994"
    if y < 2000:
        return "1995–1999"
    if y < 2005:
        return "2000–2004"
    return "2005 or later"


# (chart title, column-name substrings to coalesce, normalizer, multi-select?)
DEMOGRAPHICS: list[tuple[str, tuple[str, ...], callable, bool]] = [
    ("Pronouns", ("Pronombres", "Pronomes", "Pronouns"), norm_pronouns, False),
    ("Education", ("Educación:", "Educação:", "Education:"), norm_education, False),
    ("Field(s) — multi-select", ("Campo(s)", "Field(s)"), norm_field, True),
    ("Year of birth", ("Año de nacimiento", "Ano de nascimento", "Year of birth"), year_bucket, False),
    ("NLP level", ("nivel de conocimiento de PLN", "NLP level", "nível de conhecimento em PLN"), norm_nlp, False),
    ("First SomosNLP event?", ("primera vez participando",), norm_first_time, False),
]


# --- small plotting / data helpers -----------------------------------------


def _label_bars(ax, bars) -> None:
    """Annotate each non-zero bar with its integer height."""
    for bar in bars:
        h = bar.get_height()
        if h:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _cat_label(col: str, value: str) -> str:
    if value == "":
        return "—"
    return country_display(value) if col == "country" else value


def _rotate_xticks(ax) -> None:
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")


def _coalesce(frame: pd.DataFrame) -> pd.Series:
    """First non-empty value across the (language-variant) columns of ``frame``."""
    filled = frame.fillna("").astype(str)
    return filled.apply(
        lambda row: next((v.strip() for v in row if v.strip()), ""), axis=1
    )


def _demo_counts(series: pd.Series, normalizer, multi: bool) -> pd.Series:
    """Normalized value counts for one demographic question. Unanswered rows
    (and unparseable answers) land in a ``(no answer)`` bucket."""
    out: list[str] = []
    for raw in series:
        text = "" if raw is None else str(raw).strip()
        if not text or text.lower() == "nan":
            out.append("(no answer)")
            continue
        if multi:
            parts = [p.strip() for p in text.split("|") if p.strip()]
            if parts:
                out.extend(normalizer(p) for p in parts)
            else:
                out.append("(no answer)")
        else:
            out.append(normalizer(text))
    return pd.Series(out, dtype="object").value_counts()


def _bar_question(ax, df: pd.DataFrame, col: str) -> None:
    counts = df[col].fillna("").value_counts()
    labels = [_cat_label(col, v) for v in counts.index]
    bars = ax.bar(labels, counts.values, color=C_SENT)
    _label_bars(ax, bars)
    ax.set_title(f"by {col}")
    ax.set_ylabel("participants")
    _rotate_xticks(ax)


def _pie(ax, counts: pd.Series, title: str) -> None:
    counts = counts.sort_values(ascending=False)
    # Percentages sit inside the wedges; category names go in a legend below
    # the pie so many small slices don't pile their labels on top of each other.
    wedges, _texts, _autos = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 6 else "",
        startangle=90,
        textprops={"fontsize": 8, "color": "white", "fontweight": "bold"},
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [f"{label}  ({count})" for label, count in counts.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7.5,
        frameon=False,
        ncol=2,
    )
    ax.set_title(title, fontsize=11)


# --- data assembly ----------------------------------------------------------


def merge_demographics(
    participants: pd.DataFrame, csv_path: Path | None
) -> tuple[pd.DataFrame, list[str]]:
    """Join the Eventbrite demographic answers onto the participants table by
    HF username. Returns ``(merged_df, available_question_labels)`` — a
    question is "available" only if its columns were found and someone
    answered.

    Earlier versions joined on email, but email is no longer pushed to the
    Hub (privacy). The HF username lives in both sources, so we derive the
    cleaned username from the Eventbrite columns the same way
    ``import_participants_info`` does and use that as the key."""
    if csv_path is None or not Path(csv_path).exists():
        return participants, []

    raw = pd.read_csv(csv_path)
    if "Ticket Type" not in raw.columns:
        print(f"  (skipping demographics: {csv_path} is not an Eventbrite report)")
        return participants, []
    hack = raw[raw["Ticket Type"].str.startswith("Hackathon", na=False)].copy()
    hack["Order Date"] = pd.to_datetime(hack["Order Date"], errors="coerce", utc=True)

    # Replicate the importer's HF-username extraction so the join key here
    # matches what landed in the participants dataset.
    hf_cols = _order_by_lang(
        [c for c in hack.columns if "Hugging Face" in c or "HuggingFace" in c]
    )
    pick_first = lambda r: next((str(v).strip() for v in r if str(v).strip()), "")
    hack["_username"] = (
        hack[hf_cols].fillna("").apply(pick_first, axis=1).apply(clean_username)
    )

    available: list[str] = []
    for label, needles, _norm, _multi in DEMOGRAPHICS:
        matched = [c for c in hack.columns if any(n in c for n in needles)]
        hack[label] = _coalesce(hack[matched]) if matched else ""
        if (hack[label].astype(str).str.strip() != "").any():
            available.append(label)

    # One row per username, latest registration wins — same dedup rule as
    # the importer (which uses case-insensitive matching; mirror that here).
    hack["_key"] = hack["_username"].str.lower()
    hack = hack[hack["_key"] != ""]
    demo = (
        hack.sort_values("Order Date")
        .drop_duplicates("_key", keep="last")
        .set_index("_key")
    )
    labels = [d[0] for d in DEMOGRAPHICS]
    # Join on lowercased username so case differences between Eventbrite
    # answers and the canonical dataset don't break the merge.
    join_key = participants["username"].str.lower()
    merged = participants.join(demo[labels], on=join_key.rename("_key"))
    return merged, available


# --- figures ----------------------------------------------------------------


def plot_participants(df: pd.DataFrame, demo_labels: list[str], path: Path) -> None:
    panels = ["__language__", "__country__"] + demo_labels
    ncols = 3
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.2 * ncols, 5.6 * nrows), layout="constrained"
    )
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(
        f"hackathon_participants — {len(df)} participants",
        fontsize=15,
        fontweight="bold",
    )
    for ax, panel in zip(axes, panels):
        if panel == "__language__":
            _bar_question(ax, df, "language")
        elif panel == "__country__":
            _bar_question(ax, df, "country")
        else:
            _, _needles, normalizer, multi = next(
                d for d in DEMOGRAPHICS if d[0] == panel
            )
            _pie(ax, _demo_counts(df[panel], normalizer, multi), panel)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_prompts(df: pd.DataFrame, path: Path) -> None:
    rdf = ranking_df(df)
    cc = country_counts(df)
    n_annotators = len(rdf)

    fig, axd = plt.subplot_mosaic(
        [["totals", "ranking"], ["country", "country"]],
        figsize=(15, 11),
    )
    fig.suptitle(
        f"cultural_preferences — {len(df)} prompts · "
        f"{n_annotators} unique annotator{'' if n_annotators == 1 else 's'}",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Overall totals — global version of the Leaderboard's per-user plot.
    ax = axd["totals"]
    totals = [
        int(rdf["prompts sent"].sum()),
        int(rdf["prompts validated"].sum()),
        int(rdf["answers voted"].sum()),
    ]
    bars = ax.bar(
        ["prompts sent", "prompts validated", "answers voted"],
        totals,
        color=[C_SENT, C_VALIDATED, C_VOTED],
    )
    _label_bars(ax, bars)
    ax.set_title("totals across all users")
    ax.set_ylabel("count")

    # 2. Per-user ranking — top 10 annotators, stacked sent / validated / voted.
    ax = axd["ranking"]
    top = rdf.head(10)
    x = np.arange(len(top))
    sent = top["prompts sent"].to_numpy()
    validated = top["prompts validated"].to_numpy()
    voted = top["answers voted"].to_numpy()
    ax.bar(x, sent, color=C_SENT, label="prompts sent")
    ax.bar(x, validated, bottom=sent, color=C_VALIDATED, label="prompts validated")
    ax.bar(x, voted, bottom=sent + validated, color=C_VOTED, label="answers voted")
    ax.set_xticks(x)
    ax.set_xticklabels(top["username"], rotation=45, ha="right")
    ax.set_title(f"ranking — top {len(top)} of {n_annotators} annotators (stacked)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)

    # 3. Per-country validation status — the Leaderboard's stacked country plot.
    ax = axd["country"]
    labels = [country_display(c) for c in cc["country"]]
    ax.bar(labels, cc["fully_validated"], color=C_FULLY, label="fully validated")
    ax.bar(
        labels,
        cc["pending"],
        bottom=cc["fully_validated"],
        color=C_PENDING,
        label="pending",
    )
    ax.set_title("prompts by country: validated vs pending")
    ax.set_ylabel("prompts")
    _rotate_xticks(ax)
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- entry point ------------------------------------------------------------


def _default_csv() -> Path | None:
    # Match only the canonical Eventbrite export name (report-<date>T<time>.csv),
    # not sidecars like report-...-_missing_hf.csv.
    canonical = re.compile(r"^report-\d{4}-\d{2}-\d{2}T\d{4}\.csv$")
    found = sorted(
        p for p in (REPO_DIR / "reports").glob("report-*.csv")
        if canonical.match(p.name)
    )
    return found[-1] if found else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eventbrite-csv",
        type=Path,
        default=None,
        help="Eventbrite report CSV for demographics (default: newest reports/report-*.csv)",
    )
    args = parser.parse_args()
    csv_path = args.eventbrite_csv or _default_csv()

    OUT_DIR.mkdir(exist_ok=True)
    participants = load_participants_df()
    prompts = load_prompts_df()
    print(f"participants:          {len(participants):>5} rows")
    print(f"cultural_preferences:  {len(prompts):>5} rows")

    merged, demo_labels = merge_demographics(participants, csv_path)
    if csv_path and demo_labels:
        print(f"demographics:          {len(demo_labels)} questions from {csv_path}")
    else:
        print("demographics:          no Eventbrite CSV — plotting language/country only")

    p_path = OUT_DIR / "participants_overview.png"
    q_path = OUT_DIR / "prompts_overview.png"
    plot_participants(merged, demo_labels, p_path)
    plot_prompts(prompts, q_path)
    print(f"\nWrote {p_path}")
    print(f"Wrote {q_path}")


if __name__ == "__main__":
    main()
