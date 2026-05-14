"""Import hackathon participants from an Eventbrite report CSV.

Cleans the free-text HF-username and country fields the registration form
collects, dedupes by case-insensitive HF username (keeping the latest
registration), and pushes a fresh participants table to the private Hub
dataset.

Usage:
    # Dry run — print what would be pushed, write nothing.
    python import_participants_from_eventbrite.py reports/report-XXXX.csv

    # Actually push.
    python import_participants_from_eventbrite.py reports/report-XXXX.csv --push

Destructive: ``--push`` overwrites ``mariagrandury/hackathon_participants``.
"""

from __future__ import annotations

import argparse
import re
import sys

import pandas as pd
from datasets import Dataset

from data import HF_TOKEN, PARTICIPANTS_FEATURES, PARTICIPANTS_REPO

TICKET_TO_LANG = {
    "Hackathon (Español)": "es",
    "Hackathon (English)": "en",
    "Hackathon (Portugues)": "pt",
}

# (substring, 2-letter code). Lowercased substring match against the
# free-text country field. The list is ordered to disambiguate compound
# entries — checked against the first comma-separated component first, then
# the whole string.
COUNTRY_PATTERNS: list[tuple[str, str]] = [
    ("argentina", "ar"), ("bolivia", "bo"), ("brasil", "br"), ("brazil", "br"),
    ("chile", "cl"), ("colombia", "co"), ("costa rica", "cr"), ("cuba", "cu"),
    ("ecuador", "ec"), ("cuenca", "ec"), ("el salvador", "sv"),
    ("españa", "es"), ("espana", "es"), ("spain", "es"), ("andaluc", "es"),
    ("sevilla", "es"), ("madrid", "es"), ("catalu", "es"), ("catalonia", "es"),
    ("guatemala", "gt"), ("honduras", "hn"), ("tegucigalpa", "hn"),
    ("méxico", "mx"), ("mexico", "mx"), ("cdmx", "mx"), ("oaxaca", "mx"),
    ("nicaragua", "ni"), ("panam", "pa"), ("paraguay", "py"), ("asunci", "py"),
    ("perú", "pe"), ("peru", "pe"), ("chiclayo", "pe"), ("lima", "pe"), ("cusco", "pe"),
    ("portugal", "pt"), ("puerto rico", "pr"), ("uruguay", "uy"), ("venezuela", "ve"),
    ("canada", "ca"), ("canadá", "ca"), ("palestina", "ps"), ("palestine", "ps"),
]

URL_PATTERNS = [
    re.compile(r"https?://(?:www\.)?huggingface\.co/(?!organizations)([\w.\-]+)", re.I),
    re.compile(r"https?://(?:www\.)?hf\.co/(?!organizations)([\w.\-]+)", re.I),
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


def build_participants(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Ticket Type"].str.startswith("Hackathon", na=False)].copy()

    hf_cols = [c for c in df.columns if "Hugging Face" in c or "HuggingFace" in c]
    country_cols = [c for c in df.columns if c.startswith("País") or c.startswith("Country")]

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

    return (
        valid[["username", "language", "country", "gmail"]]
        .sort_values(["country", "language", "username"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Path to the Eventbrite report CSV")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to the Hub. Without this flag, runs as a dry-run.",
    )
    args = parser.parse_args()

    participants = build_participants(args.csv)

    print(f"Cleaned participants: {len(participants)}")
    print()
    print(participants.to_string())
    print()
    print("By language:", participants["language"].value_counts().to_dict())
    print("By country: ", participants["country"].value_counts(dropna=False).to_dict())

    if not args.push:
        print()
        print("Dry run. Re-run with --push to overwrite", PARTICIPANTS_REPO)
        return

    if not HF_TOKEN:
        sys.exit("HF_TOKEN not set; cannot push.")

    ds = Dataset.from_pandas(
        participants, preserve_index=False, features=PARTICIPANTS_FEATURES
    )
    ds.push_to_hub(PARTICIPANTS_REPO, private=True, token=HF_TOKEN)
    print(f"\nPushed {len(ds)} rows to {PARTICIPANTS_REPO}")


if __name__ == "__main__":
    main()
