"""Flatten and enrich the HF datasets into analysis-ready DataFrames.

These builders take whatever `load_participants_df()` / `load_prompts_df()`
return and produce the shapes each analysis needs:
 - participants enriched with Eventbrite demographics + test-derived columns
 - one row per prompt with derived length / status flags
 - long-form one-row-per-slot tables for validations and votes
 - one row per (user, attempt, question) for the entry-test analysis

Read-only: no Hub writes happen here.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import data as _data
import test_data as _test_data
from import_participants_info import _default_csv, _order_by_lang, clean_username


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


# ---------- Eventbrite report discovery -------------------------------------


def latest_eventbrite_report() -> Path | None:
    """Newest `report-YYYY-MM-DDTHHMM.csv` under `reports/`. Overridable via
    `HACKATHON_EVENTBRITE_CSV`. Falls back to `import_participants_info._default_csv`."""
    override = os.environ.get("HACKATHON_EVENTBRITE_CSV")
    if override:
        path = Path(override)
        return path if path.exists() else None
    return _default_csv()


# ---------- Eventbrite normalizers ------------------------------------------
# Clean re-implementations of the patterns in data/inspect_hf_dataset.py
# (lines 72-176). Each collapses the language variants of a multilingual
# form field to a small canonical set.


def norm_pronouns(v: str) -> str:
    v = (v or "").strip().lower()
    if not v:
        return "(no answer)"
    if v in ("él", "ele") or "he/him" in v:
        return "he / él"
    if v == "ella" or "she/her" in v:
        return "she / ella"
    if "elle" in v or "they/them" in v:
        return "they / elle"
    if "no decir" in v or "prefer not" in v:
        return "prefer not to say"
    return v[:24]


def norm_education(v: str) -> str:
    v = (v or "").strip().lower()
    if not v:
        return "(no answer)"
    if "doctor" in v:
        return "Tertiary — Doctoral"
    if "máster" in v or "maestr" in v or "magíster" in v or "master" in v:
        return "Tertiary — Master's"
    if "grado" in v or "licenc" in v or "bachelor" in v or "graduação" in v or "bacharel" in v:
        return "Tertiary — Bachelor's"
    if "ciclo corto" in v or "técnica" in v or "short-cycle" in v:
        return "Tertiary — short-cycle"
    if "post-secondary" in v or "non-tertiary" in v:
        return "Post-secondary"
    if "secundaria baja" in v or "middle school" in v or "junior high" in v:
        return "Lower secondary"
    if "secundaria alta" in v or "preparatoria" in v or "bachiller" in v or "high school" in v:
        return "Upper secondary"
    return v[:28] or "(no answer)"


def norm_field(v: str) -> str:
    v = (v or "").strip().lower()
    if not v:
        return "(no answer)"
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
    return v[:24]


def norm_nlp(v: str) -> str:
    v = (v or "").strip().lower()
    if not v:
        return "(no answer)"
    if "básico" in v or "basic" in v:
        return "Basic"
    if "intermedi" in v:
        return "Intermediate"
    if "avanzado" in v or "advanced" in v:
        return "Advanced"
    return v[:24]


def norm_first_event(v: str) -> str:
    v = (v or "").strip().lower()
    if not v:
        return "(no answer)"
    if v.startswith("s") or "primer" in v:
        return "First SomosNLP event"
    if v.startswith("n") or "anteriores" in v or "asistido" in v:
        return "Returning attendee"
    return v[:24]


def year_bucket(v: str) -> str:
    m = re.search(r"(?:19|20)\d{2}", str(v or ""))
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


# (column-name substrings to coalesce, normalizer, multi-select?, output_col)
DEMO_SPECS: list[tuple[tuple[str, ...], Callable[[str], str], bool, str]] = [
    (("Pronombres", "Pronomes", "Pronouns"), norm_pronouns, False, "pronouns_norm"),
    (("Educación:", "Educação:", "Education:"), norm_education, False, "education_norm"),
    (("Campo(s)", "Field(s)"), norm_field, True, "field_norm"),
    (
        ("Año de nacimiento", "Ano de nascimento", "Year of birth"),
        year_bucket,
        False,
        "birth_year_bucket",
    ),
    (
        ("nivel de conocimiento de PLN", "NLP level", "nível de conhecimento em PLN"),
        norm_nlp,
        False,
        "nlp_level_norm",
    ),
    (("primera vez participando",), norm_first_event, False, "first_event_norm"),
]


def _coalesce_first_nonempty(frame: pd.DataFrame) -> pd.Series:
    filled = frame.fillna("").astype(str)
    return filled.apply(
        lambda row: next((v.strip() for v in row if v.strip()), ""), axis=1
    )


def _build_demographics_table(csv_path: Path) -> pd.DataFrame:
    """One row per registered HF username (latest registration wins) with
    normalized demographic columns. Returns an empty frame if the CSV isn't
    an Eventbrite report."""
    raw = pd.read_csv(csv_path)
    if "Ticket Type" not in raw.columns:
        return pd.DataFrame()
    hack = raw[raw["Ticket Type"].str.startswith("Hackathon", na=False)].copy()
    hack["Order Date"] = pd.to_datetime(hack["Order Date"], errors="coerce", utc=True)

    hf_cols = _order_by_lang(
        [c for c in hack.columns if "Hugging Face" in c or "HuggingFace" in c]
    )
    if not hf_cols:
        return pd.DataFrame()
    hack["_username"] = (
        hack[hf_cols].fillna("")
        .apply(lambda r: next((str(v).strip() for v in r if str(v).strip()), ""), axis=1)
        .apply(clean_username)
    )

    out_cols: list[str] = []
    for needles, normalizer, multi, out_col in DEMO_SPECS:
        matched = [c for c in hack.columns if any(n in c for n in needles)]
        if not matched:
            hack[out_col] = ""
        else:
            coalesced = _coalesce_first_nonempty(hack[matched])
            if multi:
                hack[out_col] = coalesced.apply(
                    lambda text: " | ".join(
                        normalizer(p.strip()) for p in str(text).split("|") if p.strip()
                    )
                    if text else ""
                )
            else:
                hack[out_col] = coalesced.apply(normalizer)
        out_cols.append(out_col)

    hack["_key"] = hack["_username"].str.lower()
    hack = hack[hack["_key"] != ""]
    demo = (
        hack.sort_values("Order Date")
        .drop_duplicates("_key", keep="last")
        .set_index("_key")[out_cols]
    )
    return demo


# ---------- Participants ----------------------------------------------------


def _attempt_count(test_score_cell: str | None) -> int:
    scores = _data.parse_test_score(test_score_cell)
    return len(scores)


def load_participants_enriched(
    participants_df: pd.DataFrame | None = None,
    prompts_df: pd.DataFrame | None = None,
    eventbrite_csv: Path | None = None,
) -> pd.DataFrame:
    """One row per HF participant with:
      - canonical columns (username, language, country)
      - derived columns: country_display, n_test_attempts, best_test_score,
        passed_test, attempts_json
      - per-user activity: prompts_sent, prompts_validated, votes_cast
      - Eventbrite demographics (pronouns, education, field, birth_year_bucket,
        nlp_level, first_event) if a report CSV is available

    Missing arguments default to fresh loads via data.py + reports/."""
    if participants_df is None:
        participants_df = _data.load_participants_df()
    if prompts_df is None:
        prompts_df = _data.load_prompts_df()
    df = participants_df.copy()
    if df.empty:
        return df

    df["country_display"] = df["country"].apply(_data.country_display)
    df["attempts_json"] = df["test_score"].fillna("{}")
    df["n_test_attempts"] = df["test_score"].apply(_attempt_count)
    df["best_test_score"] = df["username"].apply(
        lambda u: _data.best_test_score(u, participants_df)
    )
    df["passed_test"] = df["best_test_score"] >= _data.TEST_PASS_THRESHOLD

    sent_counts = prompts_df["username"].value_counts() if not prompts_df.empty else pd.Series(dtype=int)
    val_counts = (
        _data._validator_usernames(prompts_df).value_counts()
        if not prompts_df.empty
        else pd.Series(dtype=int)
    )
    vote_counts = (
        _data._voter_usernames(prompts_df).value_counts()
        if not prompts_df.empty
        else pd.Series(dtype=int)
    )
    df["prompts_sent"] = df["username"].map(sent_counts).fillna(0).astype(int)
    df["prompts_validated"] = df["username"].map(val_counts).fillna(0).astype(int)
    df["votes_cast"] = df["username"].map(vote_counts).fillna(0).astype(int)

    csv_path = eventbrite_csv if eventbrite_csv is not None else latest_eventbrite_report()
    demo_cols = [spec[3] for spec in DEMO_SPECS]
    if csv_path and Path(csv_path).exists():
        demo = _build_demographics_table(Path(csv_path))
        if not demo.empty:
            # Merge on lowercased username so case differences between the
            # registration form and the canonical HF handle don't break the
            # join. `how="left"` keeps every participant; demographic columns
            # land as NaN when the user has no matching Eventbrite row.
            join_df = df.assign(_key=df["username"].str.lower())
            demo_reset = demo.reset_index().rename(columns={"_key": "_key"})
            df = (
                join_df.merge(demo_reset, on="_key", how="left")
                .drop(columns="_key")
            )
    for col in demo_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    return df.reset_index(drop=True)


# ---------- Prompts: flat + long forms --------------------------------------


def _word_count(s: str) -> int:
    return len(str(s or "").split())


def _safe_struct(value) -> dict:
    """`row["prompt_validation_i"]` is sometimes a dict, sometimes a Series-
    backed mapping. Coerce to plain dict with empty-string defaults."""
    if value is None:
        return {"choice": "", "username": ""}
    try:
        choice = value.get("choice", "") if hasattr(value, "get") else value["choice"]
        username = value.get("username", "") if hasattr(value, "get") else value["username"]
    except (KeyError, TypeError):
        return {"choice": "", "username": ""}
    return {"choice": str(choice or ""), "username": str(username or "")}


def _validation_slot_filled(struct: dict) -> bool:
    return bool(struct.get("username"))


def _vote_slot_filled(struct: dict) -> bool:
    return bool(struct.get("username"))


def _accept_side(choice: str) -> str:
    if choice in _data.ACCEPT_VALIDATION_CHOICES:
        return "accept"
    if choice in _data.REJECT_CHOICES:
        return "reject"
    return ""


def load_prompts_flat(prompts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per prompt, plus derived columns. Includes v0-imported rows
    with `is_v0_import=True` so the suite can show them or filter them out
    on demand."""
    if prompts_df is None:
        prompts_df = _data.load_prompts_df()
    if prompts_df.empty:
        return pd.DataFrame(
            columns=[
                "id", "username", "is_v0_import", "language", "country",
                "country_display", "prompt_chars", "prompt_words",
                "has_system_prompt", "system_prompt_chars",
                "validation_status", "accept_count", "reject_count",
                "is_fully_validated", "has_answers", "n_votes",
                "winning_choice", "model_a", "model_b",
            ]
        )
    df = prompts_df.copy()

    df["is_v0_import"] = df["username"].isin(_data.EXCLUDED_USERNAMES)
    df["country_display"] = df["country"].apply(_data.country_display)
    df["prompt_chars"] = df["prompt"].fillna("").astype(str).str.len()
    df["prompt_words"] = df["prompt"].fillna("").astype(str).apply(_word_count)
    df["has_system_prompt"] = df["system_prompt"].fillna("").astype(str).str.strip().ne("")
    df["system_prompt_chars"] = (
        df["system_prompt"].fillna("").astype(str).str.len()
    )

    def _agg_validations(row) -> pd.Series:
        choices = [_safe_struct(row[f"prompt_validation_{i}"])["choice"] for i in (1, 2, 3)]
        usernames = [_safe_struct(row[f"prompt_validation_{i}"])["username"] for i in (1, 2, 3)]
        filled = sum(1 for u in usernames if u)
        accept = sum(1 for c in choices if c in _data.ACCEPT_VALIDATION_CHOICES)
        reject = sum(1 for c in choices if c in _data.REJECT_CHOICES)
        return pd.Series({
            "validation_status": filled,
            "accept_count": accept,
            "reject_count": reject,
        })

    df[["validation_status", "accept_count", "reject_count"]] = df.apply(
        _agg_validations, axis=1
    )
    df["is_fully_validated"] = _data._fully_validated_mask(df)
    df["has_answers"] = df.apply(_data.has_answers, axis=1)

    def _agg_votes(row) -> pd.Series:
        votes = [_safe_struct(row[f"answer_chosen_{i}"]) for i in (1, 2, 3)]
        filled = sum(1 for v in votes if v["username"])
        if filled == 0:
            return pd.Series({"n_votes": 0, "winning_choice": ""})
        choices = [v["choice"] for v in votes if v["username"]]
        c = Counter(choices)
        winner = c.most_common(1)[0][0] if c else ""
        return pd.Series({"n_votes": filled, "winning_choice": winner})

    df[["n_votes", "winning_choice"]] = df.apply(_agg_votes, axis=1)

    return df


def load_validations_long(prompts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per FILLED validation slot. Columns:
        prompt_id, slot, validator_username, choice, side,
        prompt_author, prompt_language, prompt_country, prompt_country_display,
        prompt_chars, is_v0_prompt
    Excludes rows where no validator has acted yet."""
    if prompts_df is None:
        prompts_df = _data.load_prompts_df()
    if prompts_df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_id", "slot", "validator_username", "choice", "side",
                "prompt_author", "prompt_language", "prompt_country",
                "prompt_country_display", "prompt_chars", "is_v0_prompt",
            ]
        )
    rows: list[dict] = []
    for _, row in prompts_df.iterrows():
        for slot in (1, 2, 3):
            struct = _safe_struct(row[f"prompt_validation_{slot}"])
            if not struct["username"]:
                continue
            rows.append({
                "prompt_id": int(row["id"]),
                "slot": slot,
                "validator_username": struct["username"],
                "choice": struct["choice"],
                "side": _accept_side(struct["choice"]),
                "prompt_author": str(row["username"]),
                "prompt_language": str(row.get("language", "")),
                "prompt_country": str(row.get("country", "")),
                "prompt_country_display": _data.country_display(row.get("country")),
                "prompt_chars": len(str(row.get("prompt") or "")),
                "is_v0_prompt": row["username"] in _data.EXCLUDED_USERNAMES,
            })
    return pd.DataFrame(rows)


def _primary_validation_bucket(row) -> str:
    accepts = [
        _safe_struct(row[f"prompt_validation_{i}"])["choice"]
        for i in (1, 2, 3)
        if _safe_struct(row[f"prompt_validation_{i}"])["choice"] in _data.ACCEPT_VALIDATION_CHOICES
    ]
    if not accepts:
        return ""
    c = Counter(accepts)
    return c.most_common(1)[0][0]


def load_votes_long(prompts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per FILLED vote slot. Columns:
        prompt_id, slot, voter_username, choice, model_a, model_b,
        chosen_model, prompt_language, prompt_country,
        prompt_country_display, primary_validation_bucket"""
    if prompts_df is None:
        prompts_df = _data.load_prompts_df()
    if prompts_df.empty:
        return pd.DataFrame(
            columns=[
                "prompt_id", "slot", "voter_username", "choice",
                "model_a", "model_b", "chosen_model", "prompt_language",
                "prompt_country", "prompt_country_display",
                "primary_validation_bucket",
            ]
        )
    rows: list[dict] = []
    for _, row in prompts_df.iterrows():
        model_a = str(row.get("model_a") or "")
        model_b = str(row.get("model_b") or "")
        primary = _primary_validation_bucket(row)
        for slot in (1, 2, 3):
            struct = _safe_struct(row[f"answer_chosen_{slot}"])
            if not struct["username"]:
                continue
            choice = struct["choice"]
            if choice == "a":
                chosen = model_a
            elif choice == "b":
                chosen = model_b
            elif choice == "both":
                chosen = "both"
            elif choice == "none":
                chosen = "none"
            else:
                chosen = ""
            rows.append({
                "prompt_id": int(row["id"]),
                "slot": slot,
                "voter_username": struct["username"],
                "choice": choice,
                "model_a": model_a,
                "model_b": model_b,
                "chosen_model": chosen,
                "prompt_language": str(row.get("language", "")),
                "prompt_country": str(row.get("country", "")),
                "prompt_country_display": _data.country_display(row.get("country")),
                "primary_validation_bucket": primary,
            })
    return pd.DataFrame(rows)


# ---------- Test responses --------------------------------------------------


def _category_for_question(qid: str, question_lookup: dict[str, dict]) -> tuple[str, str]:
    q = question_lookup.get(qid)
    if q is None:
        return "", ""
    return q.get("correct_key", ""), q.get("format", "")


def _score_one(q: dict, chosen: str | None) -> tuple[float, bool]:
    if q is None:
        return 0.0, False
    fmt = q.get("format", "")
    if fmt == _test_data.SUPPORTED_FORMAT:
        score = _test_data._score_classification(q, chosen)
        return score, score == _test_data._CLASSIFICATION_EXACT
    if fmt == "multiple_choice":
        score = _test_data._score_mcq(q, chosen)
        return score, score == _test_data._MCQ_CORRECT
    return 0.0, False


def load_test_responses_long(participants_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (user, attempt, question). Columns:
        username, language, attempt, question_id, category, format,
        chosen, correct_key, is_correct_strict, points_earned
    `is_correct_strict` is True only on exact-bucket / exact-MCQ matches;
    `points_earned` uses the partial-credit scheme from test_data."""
    if participants_df is None:
        participants_df = _data.load_participants_df()
    if participants_df.empty:
        return pd.DataFrame(
            columns=[
                "username", "language", "attempt", "question_id", "category",
                "format", "chosen", "correct_key", "is_correct_strict",
                "points_earned",
            ]
        )

    # Cache question lookups per language to avoid reloading the bank each row.
    bank_cache: dict[str, dict[str, dict]] = {}

    def _lookup(lang: str) -> dict[str, dict]:
        if lang in bank_cache:
            return bank_cache[lang]
        lookup: dict[str, dict] = {}
        for q in _test_data.load_questions(lang):
            lookup[q["id"]] = q
        for q in _test_data.load_mcq_questions(lang):
            lookup[q["id"]] = q
        # Also include hidden classification questions — older attempts may
        # reference them by id.
        for q in _test_data.load_hidden_questions(lang):
            lookup.setdefault(q["id"], q)
        bank_cache[lang] = lookup
        return lookup

    rows: list[dict] = []
    for _, prow in participants_df.iterrows():
        username = str(prow.get("username") or "")
        lang = str(prow.get("language") or "es")
        responses = _data.parse_test_responses(prow.get("test_responses"))
        if not responses:
            continue
        try:
            lookup = _lookup(lang)
        except Exception:
            lookup = {}
        for attempt_str, answers in responses.items():
            try:
                attempt = int(attempt_str)
            except (ValueError, TypeError):
                continue
            for qid, chosen in answers.items():
                q = lookup.get(qid)
                category, fmt = _category_for_question(qid, lookup)
                points, is_correct = _score_one(q, chosen)
                rows.append({
                    "username": username,
                    "language": lang,
                    "attempt": attempt,
                    "question_id": qid,
                    "category": category,
                    "format": fmt,
                    "chosen": str(chosen or ""),
                    "correct_key": q.get("correct_key", "") if q else "",
                    "is_correct_strict": bool(is_correct),
                    "points_earned": float(points),
                })
    return pd.DataFrame(rows)
