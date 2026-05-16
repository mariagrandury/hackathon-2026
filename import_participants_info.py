"""Import hackathon participants from an Eventbrite report CSV.

Cleans the free-text HF-username and country fields the registration form
collects, dedupes by case-insensitive HF username (keeping the latest
registration), and pushes a fresh participants table to the private Hub
dataset. When a registrant answered the form in more than one language,
the Spanish answer wins, then Portuguese, then English (see _order_by_lang).

Both modes also write ``<input>_missing_hf.csv`` next to the input CSV: the
name + email of every attendee whose HF username was blank or unrecognized,
so organisers can follow up and get it fixed.

Usage:
    # Dry run against the newest reports/report-*.csv — print the cleaned
    # table + write the missing-username report; push nothing to the Hub.
    python import_participants_info.py

    # Pin to a specific export instead of the newest.
    python import_participants_info.py reports/report-XXXX.csv

    # Push to the Hub (uses the newest report by default).
    python import_participants_info.py --push

Destructive: ``--push`` overwrites ``mariagrandury/hackathon_participants``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset

from data import (
    EMPTY_TEST_RESPONSES,
    EMPTY_TEST_SCORE,
    HF_TOKEN,
    PARTICIPANTS_FEATURES,
    PARTICIPANTS_REPO,
    _cache_invalidate,
    load_participants_df,
)

# Repo root (this file lives at the top level). ``data/inspect_hf_dataset.py``
# imports ``_default_csv`` from here, so the lookup is consistent across
# scripts and tests.
REPO_DIR = Path(__file__).resolve().parent
REPORTS_DIR = REPO_DIR / "reports"


def _default_csv() -> Path | None:
    """Newest canonical Eventbrite export under ``reports/``.

    Matches only ``report-<YYYY-MM-DD>T<HHMM>.csv`` — skips sidecars like
    ``report-...-_missing_hf.csv`` and anything else that happens to start
    with ``report-``. Returns None if no canonical export exists yet."""
    canonical = re.compile(r"^report-\d{4}-\d{2}-\d{2}T\d{4}\.csv$")
    found = sorted(
        p for p in REPORTS_DIR.glob("report-*.csv") if canonical.match(p.name)
    )
    return found[-1] if found else None

TICKET_TO_LANG = {
    "Hackathon (Español)": "es",
    "Hackathon (English)": "en",
    "Hackathon (Portugues)": "pt",
}

# (substring, 2-letter code). Lowercased substring match against the
# free-text country field. The list is ordered to disambiguate compound
# entries — checked against the first comma-separated component first, then
# the whole string.
#
# Only the countries seen in registrations so far are listed. Anything not
# matched here (e.g. "USA", "United Kingdom") maps to "" — the row is still
# imported, just with a blank country. build_participants() prints a warning
# listing any unmatched non-empty values; add a pattern + re-run to fix one.
COUNTRY_PATTERNS: list[tuple[str, str]] = [
    ("argentina", "ar"),
    ("bolivia", "bo"),
    ("brasil", "br"),
    ("brazil", "br"),
    ("chile", "cl"),
    ("colombia", "co"),
    ("costa rica", "cr"),
    ("cuba", "cu"),
    ("ecuador", "ec"),
    ("cuenca", "ec"),
    ("el salvador", "sv"),
    ("españa", "es"),
    ("espana", "es"),
    ("spain", "es"),
    ("andaluc", "es"),
    ("sevilla", "es"),
    ("madrid", "es"),
    ("catalu", "es"),
    ("catalonia", "es"),
    ("guatemala", "gt"),
    ("honduras", "hn"),
    ("tegucigalpa", "hn"),
    ("méxico", "mx"),
    ("mexico", "mx"),
    ("cdmx", "mx"),
    ("oaxaca", "mx"),
    ("nicaragua", "ni"),
    ("panam", "pa"),
    ("paraguay", "py"),
    ("asunci", "py"),
    ("perú", "pe"),
    ("peru", "pe"),
    ("chiclayo", "pe"),
    ("lima", "pe"),
    ("cusco", "pe"),
    ("portugal", "pt"),
    ("puerto rico", "pr"),
    ("uruguay", "uy"),
    ("venezuela", "ve"),
    ("canada", "ca"),
    ("canadá", "ca"),
    ("palestina", "ps"),
    ("palestine", "ps"),
]

# HF profile URLs people paste instead of a bare username. The scheme is
# optional so "huggingface.co/foo" is caught as well as "https://...".
URL_PATTERNS = [
    re.compile(
        r"(?:https?://)?(?:www\.)?huggingface\.co/(?!organizations)([\w.\-]+)", re.I
    ),
    re.compile(r"(?:https?://)?(?:www\.)?hf\.co/(?!organizations)([\w.\-]+)", re.I),
]

# Values that look like usernames but aren't real ones — the org slug that
# people paste from the join-confirmation URL, and explicit opt-outs.
USERNAME_BLACKLIST = {"somosnlp-hackathon-2026", "nonneeded"}

# Free-text non-answers the form collected.
NON_ANSWERS = {"no tengo", "no", "none", "n/a", "na", "."}


def clean_username(raw: str) -> str:
    s = (raw or "").strip().strip("'\"").lstrip("@").strip()
    if not s or s.lower() in NON_ANSWERS:
        return ""
    for pat in URL_PATTERNS:
        m = pat.search(s)
        if m:
            s = m.group(1)
            break
    if "://" in s or " " in s:
        return ""
    s = s.rstrip(".,;:/")
    if not re.fullmatch(r"[\w.\-]+", s):
        return ""
    if s.lower() in USERNAME_BLACKLIST:
        return ""
    return s


def map_country(raw: str) -> str:
    s = (raw or "").strip().lower()
    if not s:
        return ""
    first = s.split(",", 1)[0].strip()
    for needle, code in COUNTRY_PATTERNS:
        if needle in first:
            return code
    for needle, code in COUNTRY_PATTERNS:
        if needle in s:
            return code
    return ""


# Eventbrite asks the HF-username and country questions once per ticket
# language, producing three columns each. Order them ES > PT > EN so a
# registrant who somehow answered in more than one language gets their
# Spanish answer first.
_LANG_ORDER = ("es", "pt", "en")
_LANG_MARKERS = {
    "es": ("cuál es tu nombre", "que representen"),
    "pt": ("qual é o seu nome", "que representem"),
    "en": ("which is your", "country or countries"),
}


def _order_by_lang(cols: list[str]) -> list[str]:
    def rank(col: str) -> int:
        c = col.lower()
        for i, lang in enumerate(_LANG_ORDER):
            if any(m in c for m in _LANG_MARKERS[lang]):
                return i
        return len(_LANG_ORDER)  # unrecognized columns sort last

    return sorted(cols, key=rank)


def build_participants(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(participants, missing)``.

    ``participants`` is the cleaned, deduped table ready to push.
    ``missing`` is one row per Hackathon attendee whose HF username was
    blank or unrecognized — name + email so organisers can follow up.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Ticket Type"].str.startswith("Hackathon", na=False)].copy()

    hf_cols = _order_by_lang(
        [c for c in df.columns if "Hugging Face" in c or "HuggingFace" in c]
    )
    country_cols = _order_by_lang(
        [c for c in df.columns if c.startswith("País") or c.startswith("Country")]
    )

    # First non-empty answer wins; hf_cols / country_cols are ES > PT > EN.
    pick_first = lambda r: next((str(v).strip() for v in r if str(v).strip()), "")
    df["hf_raw"] = df[hf_cols].fillna("").apply(pick_first, axis=1)
    df["country_raw"] = df[country_cols].fillna("").apply(pick_first, axis=1)
    df["username"] = df["hf_raw"].apply(clean_username)
    df["language"] = df["Ticket Type"].map(TICKET_TO_LANG).fillna("")
    df["country"] = df["country_raw"].apply(map_country)
    df["gmail"] = df["Email"].fillna("").astype(str).str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", utc=True)

    valid = df[df["username"] != ""].copy()
    valid["_key"] = valid["username"].str.lower()
    valid = valid.sort_values("Order Date").drop_duplicates("_key", keep="last")

    unmapped = sorted(
        set(
            valid.loc[
                (valid["country"] == "") & (valid["country_raw"] != ""),
                "country_raw",
            ]
        )
    )
    if unmapped:
        bar = "!" * 72
        print()
        print(bar)
        print(
            f"!! WARNING: {len(unmapped)} country value(s) not in COUNTRY_PATTERNS."
        )
        print("!! These rows were imported with a blank country, so participants")
        print("!! will see ALL prompts in the validation/voting tabs (no country")
        print("!! filter applies). To fix: add a substring pattern for each entry")
        print("!! below to COUNTRY_PATTERNS in import_participants_info.py and")
        print("!! re-run with --push.")
        print("!!")
        for v in unmapped:
            print(f"!!   - {v!r}")
        print(bar)
        print()
    else:
        print(
            f"OK: every non-blank country answer matched COUNTRY_PATTERNS "
            f"({len(valid)} participant(s))."
        )

    # ``gmail`` stays in ``valid`` for the missing-HF sidecar / local debugging
    # but is intentionally NOT included in the participants table that gets
    # pushed to the Hub — emails shouldn't live in a shared dataset.
    participants = (
        valid[["username", "language", "country"]]
        .sort_values(["country", "language", "username"])
        .reset_index(drop=True)
    )

    # "missing"          — form filled, HF username left blank
    # "unrecognized"     — form filled, HF answer isn't a usable username
    # "no_attendee_info" — Eventbrite order whose attendee details were
    #                      never filled in ("Info Requested" placeholder);
    #                      no real name/email, so not contactable from here.
    missing = df[df["username"] == ""].copy()

    def _reason(row) -> str:
        if str(row["Email"]).strip() == "Info Requested":
            return "no_attendee_info"
        return "missing" if not str(row["hf_raw"]).strip() else "unrecognized"

    missing["reason"] = missing.apply(_reason, axis=1)
    missing = (
        missing[["First Name", "Last Name", "Email", "Ticket Type", "hf_raw", "reason"]]
        .rename(
            columns={
                "First Name": "first_name",
                "Last Name": "last_name",
                "Email": "email",
                "Ticket Type": "ticket_type",
                "hf_raw": "hf_username_raw",
            }
        )
        .sort_values(["reason", "last_name", "first_name"])
        .reset_index(drop=True)
    )

    return participants, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default=None,
        help=(
            "Path to the Eventbrite report CSV. Optional: defaults to the "
            "newest reports/report-*.csv export."
        ),
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to the Hub. Without this flag, runs as a dry-run.",
    )
    args = parser.parse_args()

    csv_path = args.csv or _default_csv()
    if csv_path is None:
        sys.exit(
            f"No CSV given and no canonical export found under {REPORTS_DIR}. "
            f"Drop an Eventbrite report (report-<YYYY-MM-DD>T<HHMM>.csv) "
            f"there or pass a path explicitly."
        )
    if not args.csv:
        print(f"Using newest report: {csv_path}")

    participants, missing = build_participants(str(csv_path))

    print(f"Cleaned participants: {len(participants)}")
    print()
    print(participants.to_string())
    print()
    print("By language:", participants["language"].value_counts().to_dict())
    print("By country: ", participants["country"].value_counts(dropna=False).to_dict())

    missing_path = csv_path.with_name(f"{csv_path.stem}_missing_hf.csv")
    missing.to_csv(missing_path, index=False)
    print(
        f"\nWrote {len(missing)} attendee(s) with a missing/unrecognized HF "
        f"username to {missing_path}"
    )

    if not args.push:
        print()
        print("Dry run. Re-run with --push to overwrite", PARTICIPANTS_REPO)
        return

    if not HF_TOKEN:
        sys.exit("HF_TOKEN not set; cannot push.")

    # Preserve existing test_score AND test_responses values: re-importing
    # should refresh the participant list (new registrations, country fixes,
    # …) without wiping out anything already recorded. Unknown /
    # newly-registered usernames start with the empty sentinels. Defensive:
    # read fresh (bypass our in-process cache), and abort if scores would
    # be wiped — silent fallback is how scores got wiped once.
    _cache_invalidate(PARTICIPANTS_REPO)
    existing = load_participants_df()
    if "test_score" not in existing.columns:
        print(
            f"  existing dataset has no test_score column; "
            f"all rows start at {EMPTY_TEST_SCORE!r}"
        )
        existing_scores: dict[str, str] = {}
    else:
        existing_scores = dict(
            zip(
                existing["username"],
                existing["test_score"].fillna(EMPTY_TEST_SCORE),
            )
        )
    if "test_responses" not in existing.columns:
        print(
            f"  existing dataset has no test_responses column; "
            f"all rows start at {EMPTY_TEST_RESPONSES!r}"
        )
        existing_responses: dict[str, str] = {}
    else:
        existing_responses = dict(
            zip(
                existing["username"],
                existing["test_responses"].fillna(EMPTY_TEST_RESPONSES),
            )
        )
    # Match case-insensitively too, since the Eventbrite form lets people
    # type their HF handle in any case while the canonical dataset stores
    # whatever case the latest registration used.
    existing_scores_ci = {k.lower(): v for k, v in existing_scores.items() if k}
    existing_responses_ci = {k.lower(): v for k, v in existing_responses.items() if k}
    non_empty_scores = sum(
        1 for v in existing_scores.values() if v and v != EMPTY_TEST_SCORE
    )
    non_empty_responses = sum(
        1 for v in existing_responses.values() if v and v != EMPTY_TEST_RESPONSES
    )
    print(
        f"  preserving {non_empty_scores} non-empty test_score(s) "
        f"and {non_empty_responses} non-empty test_responses "
        f"across {len(existing_scores)} existing user(s)"
    )

    def _lookup(name: str, by_exact: dict, by_lower: dict, empty: str) -> str:
        return by_exact.get(name) or by_lower.get(name.lower()) or empty

    participants["test_score"] = participants["username"].map(
        lambda u: _lookup(u, existing_scores, existing_scores_ci, EMPTY_TEST_SCORE)
    )
    participants["test_responses"] = participants["username"].map(
        lambda u: _lookup(
            u, existing_responses, existing_responses_ci, EMPTY_TEST_RESPONSES
        )
    )
    preserved_scores = sum(
        1 for v in participants["test_score"] if v and v != EMPTY_TEST_SCORE
    )
    preserved_responses = sum(
        1 for v in participants["test_responses"] if v and v != EMPTY_TEST_RESPONSES
    )

    # Sanity guard: if the existing dataset had real scores but we're about
    # to push without any of them surviving, refuse — something's wrong with
    # the read or the case-matching, and the right move is to investigate,
    # not to clobber. Responses follow the same guard (they only exist for
    # users who have at least one score).
    if non_empty_scores > 0 and preserved_scores == 0:
        sys.exit(
            f"REFUSING TO PUSH: read {non_empty_scores} non-empty test_score(s) "
            f"from existing dataset, but 0 of them matched usernames in the new "
            f"import. Push would wipe real scores. Re-run after investigating "
            f"the username diff (likely a case-sensitivity or normalization issue)."
        )
    if non_empty_scores > 0 and preserved_scores < non_empty_scores:
        import time as _time
        print(
            f"  WARNING: {non_empty_scores - preserved_scores} existing score(s) "
            f"will be dropped (usernames in existing dataset not present in "
            f"the new import). Press Ctrl-C in the next 5s to abort."
        )
        _time.sleep(5)

    print(
        f"  pushing with {preserved_scores} preserved test_score(s) "
        f"and {preserved_responses} preserved test_responses"
    )

    ds = Dataset.from_pandas(
        participants, preserve_index=False, features=PARTICIPANTS_FEATURES
    )
    ds.push_to_hub(PARTICIPANTS_REPO, private=True, token=HF_TOKEN)
    print(f"\nPushed {len(ds)} rows to {PARTICIPANTS_REPO}")


if __name__ == "__main__":
    main()
