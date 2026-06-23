"""Let Claude annotate ``cultural_preferences`` the same way humans do.

This file owns the **Claude-only** annotation layer, kept deliberately separate
from ``data.py`` (which is the human/shared data layer) so human and Claude
logic never tangle. The *judgment* half of the flow is done by Claude Code
itself (the ``/annotate-prompts`` skill), so the labels are billed against your
Claude subscription, not the Anthropic API. The split keeps HF I/O out of the
model loop:

    1. ``--dump-pending``  → write a JSON work-list of prompts Claude hasn't
       labelled yet (merged prompt text; answer pair for the votable ones).
    2. Claude reads that file and, following the SAME guidelines humans use
       (``guidelines/guidelines_en.md`` §1.5/§3/§4), assigns each prompt a
       validation bucket, a blind A/B vote (when it has answers), the region/
       city it mentions, and a cultural topic.
    3. ``--write``         → upsert Claude's decisions into the separate private
       ``cultural_preferences_claude`` dataset (keyed by prompt ``id``).
    4. ``--report``        → counts + Claude-vs-human agreement where humans
       have caught up.

IMPORTANT: every write here targets ``CLAUDE_REPO`` only. This module must NEVER
write to the human ``cultural_preferences`` / ``hackathon_participants`` datasets
— Claude is a measured baseline, not a participant, and the human consensus and
voting gate must stay untouched.

Usage:
    python annotate_claude.py --dump-pending --out /tmp/pending.json [--limit N] \
        [--tasks both|validation|voting]
    python annotate_claude.py --write /tmp/results.json [--model claude-opus-4-8]
    python annotate_claude.py --report
    python annotate_claude.py --usage
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, Features, Value

import data

# ---------------------------------------------------------------------------
# Claude annotation dataset (parallel model-baseline, separate from the
# human cultural_preferences dataset)
# ---------------------------------------------------------------------------
#
# A separate private dataset where Claude records the *same* judgments the human
# annotators make — a validation bucket and a blind A/B vote — plus two extra
# descriptive columns only Claude fills (region/city + cultural topic). Keeping
# it separate keeps the human consensus (and the ``is_fully_validated`` voting
# gate) pure, and lets us measure Claude-vs-human agreement per cultural
# dimension once humans catch up. Rows join back to ``cultural_preferences`` on
# ``id`` (1-indexed, unique). Single-writer (this script), so a plain overwrite
# ``push_to_hub`` is enough; no CAS needed.
CLAUDE_REPO = "mariagrandury/cultural_preferences_claude"

# Cultural-topic taxonomy. AlKhamissi et al. (2025, arXiv:2510.05931) defines
# the four *dimensions* (knowledge/preference/dynamics/bias) the validation
# buckets already use, but deliberately gives no *topic* taxonomy and warns
# against reusing value-dimension schemes (Hofstede/GLOBE) as content buckets.
# So these topic keys are grounded in the benchmarks that paper surveys —
# BLEnD (food/family/holidays/work/sports), CANDLE (material-culture facets +
# its occupation axis, which is what the shepherd/fisher role-plays fall under),
# CultureAtlas (etiquette/greetings/religion) — plus the Adilazuarda et al.
# (2024) "Culturally Aware NLP" survey for language/arts/values. Single-label;
# pick the predominant facet. ``other`` is the catch-all.
CULTURAL_TOPICS = (
    "food_and_drink",
    "festivals_and_celebrations",
    "religion_and_beliefs",
    "family_and_kinship",
    "social_norms_and_etiquette",
    "work_and_occupations",
    "language_and_dialect",
    "arts_music_dance",
    "clothing_and_appearance",
    "traditions_and_customs",
    "geography_and_local_life",
    "history_and_heritage",
    "values_and_opinions",
    "sports_and_leisure",
    "other",
)
CULTURAL_TOPIC_SET = frozenset(CULTURAL_TOPICS)

CLAUDE_FEATURES = Features(
    {
        "id": Value("int64"),  # join key → cultural_preferences.id
        "validation_choice": Value("string"),  # one of data.VALIDATION_CHOICES, or ""
        "validation_reason": Value("string"),
        "vote_choice": Value("string"),  # one of VOTE_CHOICES_CLAUDE, or ""
        "vote_reason": Value("string"),
        "region": Value("string"),  # city/region/comarca mentioned, or ""
        "cultural_topic": Value("string"),  # one of CULTURAL_TOPICS, or ""
        "model": Value("string"),  # which Claude model produced the labels
        "labeled_at": Value("string"),  # ISO-8601 UTC timestamp
    }
)

# Mirror of app.py's VOTE_CHOICES — duplicated (not imported) because neither
# this module nor data.py may depend on the Gradio layer. Kept in sync by the
# test_claude_annotations cross-check against app.py.
VOTE_CHOICES_CLAUDE = ("a", "b", "both", "none")

_EMPTY_CLAUDE_RECORD = {
    "validation_choice": "",
    "validation_reason": "",
    "vote_choice": "",
    "vote_reason": "",
    "region": "",
    "cultural_topic": "",
    "model": "",
    "labeled_at": "",
}

# Validation buckets / vote choices / topics Claude is allowed to emit — used
# to validate ``--write`` input before it touches the Hub.
VALID_VALIDATION = set(data.VALIDATION_CHOICES)
VALID_VOTE = set(VOTE_CHOICES_CLAUDE)

DEFAULT_MODEL = "claude-opus-4-8"

# Append-only usage ledger. Every ``--write`` records how many prompts/calls
# and how many tokens a labelling pass cost, so we can extrapolate cost when
# more prompts arrive. Tokens are *measured* when the orchestrator passes them
# (e.g. from a Workflow's budget), else *estimated* from payload size.
RUNS_LOG = Path(__file__).resolve().parent / "annotate_runs.jsonl"

# Rough per-call fixed overhead: each labelling agent reads the validation +
# voting rubric (guidelines §1.5/§3/§4) before judging. Estimate in tokens;
# used only when the orchestrator doesn't supply measured token counts.
GUIDELINES_TOKENS_EST = 2500


def empty_claude_df() -> "data.pd.DataFrame":
    return data.pd.DataFrame(columns=list(CLAUDE_FEATURES))


def load_claude_df() -> "data.pd.DataFrame":
    """Read the Claude-annotations dataset, or an empty correctly-shaped frame
    if it doesn't exist yet (first run, before any annotation is written)."""
    try:
        return data._fetch_dataset(CLAUDE_REPO)
    except Exception:  # repo missing / first run / offline
        return empty_claude_df()


def _nonempty_ids(claude_df, column: str) -> set[int]:
    """Ids whose ``column`` has already been filled in the Claude dataset."""
    if claude_df.empty or column not in claude_df.columns:
        return set()
    done = claude_df[claude_df[column].astype(str).str.strip() != ""]
    return {int(i) for i in done["id"]}


def pending_validation_ids(prompts_df, claude_df) -> list[int]:
    """Prompt ids Claude hasn't validated yet (every prompt is validatable)."""
    done = _nonempty_ids(claude_df, "validation_choice")
    return [int(i) for i in prompts_df["id"] if int(i) not in done]


def pending_vote_ids(prompts_df, claude_df) -> list[int]:
    """Prompt ids that have both answers but no Claude vote yet."""
    if prompts_df.empty:
        return []
    has_ans = (prompts_df["answer_a"].astype(str).str.strip() != "") & (
        prompts_df["answer_b"].astype(str).str.strip() != ""
    )
    votable = {int(i) for i in prompts_df.loc[has_ans, "id"]}
    done = _nonempty_ids(claude_df, "vote_choice")
    return sorted(votable - done)


def write_claude_annotations(records: list[dict]) -> int:
    """Upsert annotation ``records`` (keyed by ``id``) into the Claude dataset
    and return the resulting row count.

    Each record carries an ``id`` plus any of the annotation fields. Fields left
    out of a record don't clobber what's already stored — so a validation pass
    and a later voting pass on the same ``id`` merge into one row. Creates the
    repo on first write. Writes ONLY to ``CLAUDE_REPO`` (never the human
    datasets). Single-writer, so a whole-dataset overwrite push is fine (and it
    keeps the dataset card's row count fresh, unlike the CAS path)."""
    api = data._hf_api()
    api.create_repo(
        repo_id=CLAUDE_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=data.HF_TOKEN,
    )
    existing = load_claude_df()
    by_id: dict[int, dict] = {}
    if not existing.empty:
        for r in existing.to_dict("records"):
            by_id[int(r["id"])] = {**_EMPTY_CLAUDE_RECORD, **r, "id": int(r["id"])}
    for rec in records:
        rid = int(rec["id"])
        cur = by_id.get(rid, {**_EMPTY_CLAUDE_RECORD, "id": rid})
        cur.update({k: ("" if v is None else v) for k, v in rec.items() if k != "id"})
        cur["id"] = rid
        by_id[rid] = cur
    rows = sorted(by_id.values(), key=lambda r: r["id"])
    out = data.pd.DataFrame(rows, columns=list(CLAUDE_FEATURES)).fillna("")
    out["id"] = out["id"].astype("int64")
    Dataset.from_pandas(out, preserve_index=False, features=CLAUDE_FEATURES).push_to_hub(
        CLAUDE_REPO, private=True, token=data.HF_TOKEN
    )
    return len(out)


# ---------------------------------------------------------------------------
# CLI: dump work-list → (Claude judges) → write → report
# ---------------------------------------------------------------------------


def _est_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token). Used only as a
    fallback when real token counts aren't supplied — labelled "estimated" in
    the ledger so it's never confused with a measured figure."""
    return math.ceil(len(text or "") / 4)


def _merge_prompt(system_prompt: str, prompt: str) -> str:
    """Same system+prompt merge the validation/voting tabs show annotators
    (kept in sync with app._merged_prompt_display; inlined so this data-side
    script doesn't import the Gradio layer)."""
    sm = (system_prompt or "").strip()
    p = (prompt or "").strip()
    return f"{sm}\n\n{p}" if sm else p


def dump_pending(out_path: str, *, limit: int | None, tasks: str) -> int:
    prompts_df = data.load_prompts_df()
    claude_df = load_claude_df()

    want_val = tasks in ("both", "validation")
    want_vote = tasks in ("both", "voting")
    val_ids = set(pending_validation_ids(prompts_df, claude_df)) if want_val else set()
    vote_ids = set(pending_vote_ids(prompts_df, claude_df)) if want_vote else set()

    by_id = prompts_df.set_index("id", drop=False)
    items: list[dict] = []
    for pid in sorted(val_ids | vote_ids):
        row = by_id.loc[pid]
        needs_vote = pid in vote_ids
        item = {
            "id": int(pid),
            "country": str(row.get("country", "") or ""),
            "language": str(row.get("language", "") or ""),
            "prompt": _merge_prompt(row.get("system_prompt", ""), row.get("prompt", "")),
            "needs_validation": pid in val_ids,
            "needs_vote": needs_vote,
        }
        if needs_vote:
            # Blind A/B — model names are deliberately withheld.
            item["answer_a"] = str(row.get("answer_a", "") or "")
            item["answer_b"] = str(row.get("answer_b", "") or "")
        items.append(item)

    if limit is not None:
        items = items[:limit]

    Path(out_path).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    n_val = sum(1 for it in items if it["needs_validation"])
    n_vote = sum(1 for it in items if it["needs_vote"])
    payload_tokens = sum(
        _est_tokens(it["prompt"])
        + _est_tokens(it.get("answer_a", ""))
        + _est_tokens(it.get("answer_b", ""))
        for it in items
    )
    print(
        f"wrote {len(items)} pending prompt(s) to {out_path} "
        f"({n_val} need validation, {n_vote} need a vote)"
    )
    print(f"  est. payload input tokens (prompts+answers): ~{payload_tokens:,}")
    return len(items)


def _log_run(record: dict) -> None:
    with RUNS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _estimate_run_tokens(records: list[dict], prompts_df, n_calls: int) -> tuple[int, int]:
    """Fallback token estimate for a labelling pass: input = prompt/answer text
    each labelled record covered + per-call rubric overhead; output = the JSON
    choices/reasons Claude produced."""
    by_id = prompts_df.set_index("id")
    tok_in = n_calls * GUIDELINES_TOKENS_EST
    tok_out = 0
    for rec in records:
        rid = int(rec["id"])
        if rid in by_id.index:
            row = by_id.loc[rid]
            tok_in += _est_tokens(str(row.get("system_prompt", "")))
            tok_in += _est_tokens(str(row.get("prompt", "")))
            if rec.get("vote_choice"):
                tok_in += _est_tokens(str(row.get("answer_a", "")))
                tok_in += _est_tokens(str(row.get("answer_b", "")))
        tok_out += _est_tokens(json.dumps(rec, ensure_ascii=False))
    return tok_in, tok_out


def write_results(
    in_path: str,
    *,
    model: str,
    calls: int | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    source: str = "",
) -> int:
    raw = json.loads(Path(in_path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("results file must be a JSON list of {id, ...} records")

    stamp = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for i, r in enumerate(raw):
        if "id" not in r:
            raise SystemExit(f"record {i} is missing 'id'")
        rec: dict = {"id": int(r["id"]), "model": model, "labeled_at": stamp}
        vc = (r.get("validation_choice") or "").strip()
        if vc:
            if vc not in VALID_VALIDATION:
                raise SystemExit(
                    f"record id={r['id']}: invalid validation_choice {vc!r}; "
                    f"must be one of {sorted(VALID_VALIDATION)}"
                )
            rec["validation_choice"] = vc
            rec["validation_reason"] = (r.get("validation_reason") or "").strip()
        vo = (r.get("vote_choice") or "").strip()
        if vo:
            if vo not in VALID_VOTE:
                raise SystemExit(
                    f"record id={r['id']}: invalid vote_choice {vo!r}; "
                    f"must be one of {sorted(VALID_VOTE)}"
                )
            rec["vote_choice"] = vo
            rec["vote_reason"] = (r.get("vote_reason") or "").strip()
        # Descriptive columns extracted during the validation pass.
        region = (r.get("region") or "").strip()
        if region:
            rec["region"] = region
        topic = (r.get("cultural_topic") or "").strip()
        if topic:
            if topic not in CULTURAL_TOPIC_SET:
                raise SystemExit(
                    f"record id={r['id']}: invalid cultural_topic {topic!r}; "
                    f"must be one of {sorted(CULTURAL_TOPIC_SET)}"
                )
            rec["cultural_topic"] = topic
        if not (
            "validation_choice" in rec
            or "vote_choice" in rec
            or "region" in rec
            or "cultural_topic" in rec
        ):
            continue  # nothing to record for this id
        records.append(rec)

    if not records:
        print("no usable records in results file; nothing written")
        return 0
    total = write_claude_annotations(records)

    # Usage ledger — always written, so we can extrapolate cost as more prompts
    # arrive. Prefer measured token counts (e.g. from a Workflow budget); fall
    # back to a payload-size estimate otherwise.
    n_validated = sum(1 for r in records if "validation_choice" in r)
    n_voted = sum(1 for r in records if "vote_choice" in r)
    n_calls = calls if calls is not None else 1
    measured = tokens_in is not None or tokens_out is not None
    if measured:
        tok_in, tok_out = int(tokens_in or 0), int(tokens_out or 0)
    else:
        tok_in, tok_out = _estimate_run_tokens(records, data.load_prompts_df(), n_calls)
    _log_run(
        {
            "timestamp": stamp,
            "model": model,
            "source": source,
            "prompts": len(records),
            "validated": n_validated,
            "voted": n_voted,
            "calls": n_calls,
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "tokens_total": tok_in + tok_out,
            "tokens_measured": measured,
        }
    )

    print(f"upserted {len(records)} record(s); Claude dataset now holds {total} row(s)")
    print(
        f"  logged run: {len(records)} prompts, {n_calls} call(s), "
        f"{tok_in + tok_out:,} tokens ({'measured' if measured else 'estimated'}) "
        f"-> {RUNS_LOG.name}"
    )
    return len(records)


def report() -> None:
    prompts_df = data.load_prompts_df()
    claude_df = load_claude_df()

    n_prompts = len(prompts_df)
    n_validated = len(_nonempty_ids(claude_df, "validation_choice"))
    n_voted = len(_nonempty_ids(claude_df, "vote_choice"))
    n_region = len(_nonempty_ids(claude_df, "region"))
    n_topic = len(_nonempty_ids(claude_df, "cultural_topic"))
    n_pending_val = len(pending_validation_ids(prompts_df, claude_df))
    n_pending_vote = len(pending_vote_ids(prompts_df, claude_df))

    print("=== Claude annotation coverage ===")
    print(f"prompts total:            {n_prompts}")
    print(f"validated by Claude:      {n_validated}  (pending: {n_pending_val})")
    print(f"voted by Claude:          {n_voted}  (pending: {n_pending_vote})")
    print(f"region tagged:            {n_region}")
    print(f"cultural_topic tagged:    {n_topic}")

    def _dist(column: str) -> dict:
        if claude_df.empty or column not in claude_df.columns:
            return {}
        sub = claude_df[claude_df[column].astype(str).str.strip() != ""]
        return sub[column].value_counts().to_dict()

    val_dist = _dist("validation_choice")
    if val_dist:
        print("\nClaude validation buckets:")
        for k, v in val_dist.items():
            tag = "accept" if k in data.ACCEPT_VALIDATION_CHOICES else "reject"
            print(f"  {k:12s} {v:5d}  ({tag})")

    topic_dist = _dist("cultural_topic")
    if topic_dist:
        print("\nClaude cultural topics:")
        for k, v in topic_dist.items():
            print(f"  {k:28s} {v:5d}")

    # Claude-vs-human agreement, where humans have caught up. Human validation
    # is a 3-slot consensus; only compare prompts whose three slots are ALL
    # filled (a partial consensus isn't a verdict), and compare the
    # accept/reject *verdict* (all three accept => human-accept), since the
    # fine bucket can legitimately differ between annotators.
    fv_mask = data._fully_validated_mask(prompts_df)
    human_validated = prompts_df[
        prompts_df[list(data.VALIDATION_COLS)].apply(
            lambda r: all(v["username"] for v in r), axis=1
        )
    ]
    if not human_validated.empty and not claude_df.empty:
        claude_by_id = claude_df.set_index("id")
        agree = total = 0
        for _, prow in human_validated.iterrows():
            pid = int(prow["id"])
            if pid not in claude_by_id.index:
                continue
            cv = str(claude_by_id.loc[pid, "validation_choice"]).strip()
            if not cv:
                continue
            human_accept = bool(fv_mask.loc[prow.name])
            claude_accept = cv in data.ACCEPT_VALIDATION_CHOICES
            total += 1
            agree += int(human_accept == claude_accept)
        if total:
            print(
                f"\nClaude-vs-human validation verdict agreement: "
                f"{agree}/{total} ({agree / total:.0%})"
            )
        else:
            print("\nNo overlapping validated prompts to compare yet.")
    else:
        print("\nNo human validations yet — agreement report will fill in later.")


def usage_summary() -> None:
    if not RUNS_LOG.exists():
        print("no runs logged yet")
        return
    runs = [json.loads(line) for line in RUNS_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not runs:
        print("no runs logged yet")
        return
    tot_prompts = sum(r.get("prompts", 0) for r in runs)
    tot_calls = sum(r.get("calls", 0) for r in runs)
    tot_tokens = sum(r.get("tokens_total", 0) for r in runs)
    print(f"=== usage across {len(runs)} run(s) ===")
    print(f"prompts labelled: {tot_prompts:,}")
    print(f"agent calls:      {tot_calls:,}")
    print(f"tokens (total):   {tot_tokens:,}")
    if tot_prompts:
        print(f"avg tokens/prompt: {tot_tokens / tot_prompts:,.0f}")
        remaining = max(0, len(data.load_prompts_df()) - tot_prompts)
        print(
            f"extrapolated tokens for {remaining:,} remaining validations: "
            f"~{remaining * (tot_tokens / tot_prompts):,.0f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-pending", action="store_true", help="write the work-list")
    ap.add_argument("--write", metavar="RESULTS_JSON", help="upsert Claude's decisions")
    ap.add_argument("--report", action="store_true", help="coverage + agreement")
    ap.add_argument("--usage", action="store_true", help="summarise the run ledger")
    ap.add_argument("--out", default="/tmp/annotate_pending.json", help="dump-pending output path")
    ap.add_argument("--limit", type=int, default=None, help="cap number of prompts dumped")
    ap.add_argument(
        "--tasks",
        choices=("both", "validation", "voting"),
        default="both",
        help="which work to include in the dump",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model tag stored with --write")
    # Usage instrumentation for --write (logged to annotate_runs.jsonl).
    ap.add_argument("--calls", type=int, default=None, help="number of agent calls this pass")
    ap.add_argument("--tokens-in", type=int, default=None, help="measured input tokens")
    ap.add_argument("--tokens-out", type=int, default=None, help="measured output tokens")
    ap.add_argument("--source", default="", help="label for this run (trial/full/incremental)")
    args = ap.parse_args()

    if args.dump_pending:
        dump_pending(args.out, limit=args.limit, tasks=args.tasks)
    elif args.write:
        write_results(
            args.write,
            model=args.model,
            calls=args.calls,
            tokens_in=args.tokens_in,
            tokens_out=args.tokens_out,
            source=args.source,
        )
    elif args.report:
        report()
    elif args.usage:
        usage_summary()
    else:
        ap.error("pick one of --dump-pending / --write / --report / --usage")


if __name__ == "__main__":
    main()
