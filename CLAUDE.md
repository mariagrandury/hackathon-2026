# CLAUDE.md

Guidance for AI assistants working on this repository.

## What this is

A private Hugging Face Space (Gradio SDK) for the 2026 hackathon on cultural
preferences in LLMs. Participants write prompts that probe culturally-grounded
behaviour, validate each other's prompts, and vote on the better of two model
answers. All state lives in two private datasets on the Hub.

### Architecture

- **`app.py`** — Gradio Blocks UI. Six tabs: Annotation Guidelines, Entry
  Test, Prompt Writing, Prompt Validation, Answer Voting, Leaderboard. Auth
  uses HF OAuth (`hf_oauth: true` in `README.md`); the logged-in HF username
  is the canonical identity used for every write. The Writing / Validation /
  Voting tabs are gated behind the Entry Test — `init_ui` sets them
  `visible=False` until the user's best score in `hackathon_participants`
  is ≥ `TEST_PASS_THRESHOLD` (0.95). `submit_test` reveals them in-session
  on a passing run; passing is sticky because `best_test_score` takes the
  max across attempts. The Leaderboard tab is lazy: its
  `cultural_preferences` read + plot rebuild only fires on `tab.select`,
  not on `demo.load`, so page-loads don't pay the leaderboard cost.
- **`data.py`** — pure data layer. Schemas (`PARTICIPANTS_FEATURES`,
  `PROMPTS_FEATURES`), cached reads (`load_prompts_df` /
  `load_participants_df` go through a 30 s TTL in-process cache),
  CAS write paths (`commit_prompts_with_cas`, `record_test_attempt`),
  in-memory slot reservation table (`reserve_slot` / `release_slot` /
  `is_slot_reserved_by_other`, 120 s TTL), row predicates
  (`is_fully_validated`, `has_answers`), test-score helpers
  (`best_test_score`, `parse_test_score`), and aggregations for the
  leaderboard (`user_stats`, `country_counts`, `ranking_df`). Both write
  paths share `_commit_with_cas`: each commit carries the parent SHA the
  writer was working from; on HTTP 412 the cache is invalidated and the
  mutator re-runs against the fresh state (jittered exponential backoff,
  up to `_COMMIT_MAX_RETRIES = 8`). The mutator can return `None` to
  abort (precondition no longer holds) or a mutated df to commit. Save
  handlers in `app.py` use the abort path for "already a validator on
  this row" (silent swallow) and "all three slots claimed" (slot-taken
  message). Reads use `verification_mode="no_checks"` because CAS commits
  rewrite only the parquet shard, never `README.md`, so the dataset card's
  `num_examples` goes stale after the first save; the parquet at the
  pinned `revision=sha` is authoritative.
- **`test_data.py`** — question bank + grader for the Entry Test. Reads
  `data/test-2026.json` (currently Spanish-only; other languages reuse it).
  `load_questions` returns classification questions only — `multiple_choice`
  ones are dropped because the test renders each Q as a single radio. The
  Spanish display labels in `correct` are mapped to canonical bucket keys
  (`trivial` / `stereotype` / `unrelated` / `knowledge` / `preference` /
  `dynamics` / `bias_probe`) that match `VALIDATION_CHOICES` in `app.py`.
- **`seed_datasets.py`** — one-shot script that overwrites both private
  datasets with dummy rows in the current schema. Used for local testing
  without v0 data.
- **`import_dpo_pairs.py`** — one-shot script that reads the local 2025
  dataset (`data/dataset-2025.json`, ~2136 rows grouped by country) and
  writes its prompts into `cultural_preferences` in our schema (empty
  validation/vote slots, `username` set to the `"v0"` sentinel, random
  per-row A/B assignment so voters don't see a fixed correlation with the
  2025 chosen answer; `tie`/`both_bad` rows keep the prompt but drop the
  answer pair). Use this for the real run.
- **`import_participants_info.py`** — one-shot script that turns an
  Eventbrite registration export (`reports/report-*.csv`) into the
  `hackathon_participants` dataset: cleans the free-text HF-username and
  country fields, dedupes by case-insensitive username (latest wins), and
  on `--push` overwrites the dataset. Always writes a
  `<input>_missing_hf.csv` sidecar listing attendees whose HF username was
  blank or unrecognized so organisers can chase them. Multi-language form
  columns are resolved ES > PT > EN. Before pushing it reads the existing
  dataset and preserves `test_score` per username, so a re-import to refresh
  the participant list doesn't wipe out scores already earned.
- **`guidelines.md`** — placeholder annotation guidelines, rendered as the
  first tab.
- **`requirements.txt`** — pins `gradio[oauth]==4.44.1`,
  `huggingface_hub<1.0` (Gradio 4.44 still imports the removed `HfFolder`),
  `datasets<4.0`, `pandas`. Leaderboard plots use Gradio's native `BarPlot`,
  not matplotlib.

### Datasets

Both private, owned by `mariagrandury`. Schema is the source of truth in
`data.py`; `seed_datasets.py` only re-exports it.

| Dataset                                | Columns                                                                                                                                         |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `mariagrandury/hackathon_participants` | `username`, `language`, `country`, `test_score`                                                                                                 |
| `mariagrandury/cultural_preferences`   | `username`, `language`, `country`, `prompt`, `prompt_validation_{1,2,3}`, `answer_a`, `model_a`, `answer_b`, `model_b`, `answer_chosen_{1,2,3}` |

Email is intentionally NOT in the participants schema: `import_participants_info.py`
keeps it locally (in the missing-HF sidecar CSV that organisers use to chase
attendees who didn't fill in their HF handle) but drops it before
`push_to_hub`. `data/inspect_hf_dataset.py` joins demographics by HF
username instead of by email.

`test_score` is a JSON-encoded `{attempt_number_str: score_float}` map
(e.g. `'{"1": 0.85, "2": 0.95}'`); empty sentinel is `"{}"`. It's a string
column because `datasets.Features` has no open-ended dict type, and the
dict is small. `best_test_score` returns the max across attempts, so a
later worse retry can't lock a passing user out.

Both `prompt_validation_i` and `answer_chosen_i` are `{choice: str, username: str}`.

- `answer_chosen_i.choice ∈ {"a", "b", "both", "none"}`.
- `prompt_validation_i.choice` is one of seven categories from the
  `VALIDATION_CHOICES` tuple in `app.py`, organised as **3 reject buckets**
  (`trivial`, `stereotype`, `unrelated`) and **4 accept buckets** that mirror
  the [AlKhamissi et al. (2025)](https://arxiv.org/abs/2510.05931) cultural
  dimensions: `knowledge`, `preference`, `dynamics`, `bias_probe`. The accept
  set is exposed as `data.ACCEPT_VALIDATION_CHOICES` and is what
  `is_fully_validated` checks against — a single reject from any of the three
  validators keeps the prompt out of the voting pool.

Convention for the `_i` slot columns: an empty `username` means "slot not
filled". So `{choice: "", username: ""}` is "no one has touched this slot
yet"; a slot with a reject-bucket choice is "validated as unfit".

## Behavioural rules

These are user-facing rules baked into the handlers — preserve them when
refactoring:

- The Writing / Validation / Voting tabs are **hidden until the user
  passes the Entry Test** (best score ≥ `TEST_PASS_THRESHOLD`). The gate
  is **UI-only and intentionally so** — write handlers don't re-check.
  Hackathon participants are trusted; a determined user could still
  call the Gradio API directly. Known and deferred; see the Entry-test
  follow-ups section under Future steps. If we ever need to harden it,
  add a server-side `has_passed_test` check to `save_prompt`, the
  validation save, and the vote handlers.
- A user **cannot validate their own prompt**, but **can vote on their own
  prompt** once it's fully validated.
- Validation slot picker skips prompts the user authored or already validated
  in any of the three slots.
- Voting slot picker skips prompts that are not fully validated (all three
  `prompt_validation_i.choice` in `ACCEPT_VALIDATION_CHOICES`), have no
  `answer_a`/`answer_b`, or the user already voted in any of the three slots.
- After a vote click, the next eligible prompt is loaded automatically (single
  click → save + advance).
- Leaderboard is per logged-in user. The horizontal bar plot uses a fixed goal
  of `LEADERBOARD_GOAL = 100`. The country plot is stacked: green = fully
  validated, yellow = pending; total height = total prompts sent for that
  country.

## Local development

The Space is the only place where OAuth and dataset I/O both work cleanly.
Local runs need an HF token and use Gradio's pretend-OAuth.

```bash
# 1. Clone (or pull) and enter the repo
git clone https://github.com/mariagrandury/hackathon-2026.git
cd hackathon-2026

# 2. Create a virtualenv (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned deps
pip install -r requirements.txt

# 4. Authenticate to Hugging Face — needed for both dataset I/O *and*
#    Gradio's local OAuth mock (which calls whoami() to get a fake profile).
#    Easiest: copy .env.example to .env and paste a token with read+write
#    on both private datasets above. data.py loads .env on import.
cp .env.example .env
$EDITOR .env
# Alternatives: huggingface-cli login   |   export HF_TOKEN=hf_...

# 5. Seed the dummy datasets (one-time, or after schema changes in data.py).
#    DESTRUCTIVE: this overwrites the datasets, dropping any rows users have
#    added through the Space.
python seed_datasets.py

# 6. Launch the Gradio app
python app.py
# → http://localhost:7860
```

## Tests

> **Workflow rule.** After every implementation change — whether you're
> editing a handler, a helper, a translation, the JSON bank, or the
> threshold constants — run **both** the unit suite and the integration
> script and make sure they're green before declaring the work done:
>
> ```bash
> python -m unittest discover tests   # all unit tests (fast, ~0.1s)
> python -m tests.test_integration    # end-to-end HF-stubbed (~0.7s)
> ```
>
> If a test fails because the behaviour intentionally changed, update the
> test (and document the why in the commit body) — don't skip it. New
> features land with new tests covering them.

Layout under `tests/` — all stdlib `unittest`, no extra deps:

| File | What it covers |
|------|----------------|
| `test_grading.py` | `test_data.grade` per-question scoring (classification partial-credit + MCQ ±1), `grade()` composition, bank-loader invariants. |
| `test_data_helpers.py` | `data.py` pure helpers: `parse_test_score`, `best_test_score`, `country_display`, `is_fully_validated`, `has_answers`, `_fully_validated_mask`, `user_stats`, `country_counts`, `ranking_df`, `all_known_usernames`. |
| `test_app_helpers.py` | `app.py` utilities: `_fmt_score`, `_pass_raw`, `_test_max_possible`, `_t`, `_resolve_language`, `_default_system_prompt`, `_merged_prompt_display`, `_clear_other_radio`, `show_user`, `_read_guidelines`, `_validation_reject/_accept_choices`. |
| `test_importer.py` | `import_participants_info`: `clean_username` (URL extraction, blacklist, non-answers), `map_country` (pattern matching, first-comma-component bias), `_order_by_lang` (ES > PT > EN ordering). |
| `test_test_tab.py` | Entry-test flow integration: `load_test` shape + already-passed shortcut, `submit_test` perfect / failed / unanswered / not-participant branches with `record_test_attempt` mocked out. |
| `test_record_test_attempt.py` | Commit-style update path with `HfApi` mocked: happy path with `parent_commit`, attempt-number increment, `LookupError`, retry on 412, non-412 propagates, `RuntimeError` after `_COMMIT_MAX_RETRIES`. |
| `test_integration.py` | End-to-end script with the HF I/O layer stubbed. Drives every user-facing handler, reports per-call latency, asserts the saved row shows up in the next fetch, the picker skips own prompts / already-validated rows, voting unlocks after 3 accept validations, country filter holds, commit messages carry the right action + ID. Run as a script (the bare `test_X()` functions are not discovered by `unittest`). |

Synthetic test fixtures (patched in via `unittest.mock`) keep the unit
suites independent of `data/test-2026.json`'s current contents and of the
Hub; the small `BankLoaders` section in `test_grading.py` and a couple of
`_test_max_possible` checks in `test_app_helpers.py` deliberately hit the
real JSON to lock the `TEST_QUESTIONS_PER_CATEGORY` quota.

Locally, Gradio's `LoginButton` mocks the OAuth flow with whatever HF account
your CLI / `HF_TOKEN` is logged in as, so `Save prompt` only succeeds if your
username is in the participants seed (the default seed includes
`mariagrandury`).

## Deploying to Hugging Face Spaces

The Space lives at `somosnlp-hackathon-2026/cultural-preferences` (private,
Gradio SDK). Datasets stay under `mariagrandury/` — Space and dataset owners
are intentionally different.

1. Create a private Gradio Space at `somosnlp-hackathon-2026/cultural-preferences`.
   Don't use the README template — first push from `deploy_to_space.sh`
   should land cleanly without a merge.
2. In the Space settings, add an `HF_TOKEN` secret with read+write access to
   the two private `mariagrandury/...` datasets.
3. Don't push this whole repo. The Space mirror is curated:
   `./deploy_to_space.sh` rsyncs only `app.py`, `data.py`, `test_data.py`,
   `requirements.txt`, `README.md`, `guidelines/`, `images/`, and the single
   file `data/test-2026.json` (the Entry Test question bank — the rest of
   `data/` stays out) into a sibling clone of the Space repo (default
   `../2026-space-cultural-preferences`). Commit + push from that clone.
   `CLAUDE.md`, `seed_datasets.py`, `import_dpo_pairs.py`,
   `import_participants_info.py`, `tests/`, the rest of `data/`,
   `reports/`, `.env*` deliberately stay out of the Space.

The Space auto-detects the SDK and `app_file: app.py`. OAuth Just Works
inside the Space because `hf_oauth: true` is set.

## Caveats

- **Write paths are CAS, not push_to_hub.** Both `commit_prompts_with_cas`
  and `record_test_attempt` upload only the parquet shard via
  `api.create_commit(parent_commit=sha)`; on HTTP 412 (someone else
  committed since we read) the cache is invalidated, the mutator re-runs
  against the fresh state, and we retry with jittered exponential backoff
  up to `_COMMIT_MAX_RETRIES = 8`. The mutator can return `None` to abort
  cleanly — used by the save handlers for "already a validator on this
  row" (silent swallow → "saved") and "all three slots claimed by others"
  (→ slot-taken message). Trade-off: each commit rewrites the entire
  parquet shard, and the helper assumes the dataset is one shard
  (`_single_parquet_path` raises otherwise). If either dataset ever grows
  past a single shard, that helper needs updating.
- **Read cache is per-process with 30 s TTL.** `load_prompts_df` /
  `load_participants_df` go through `_cached_load`, so back-to-back tab
  opens / pickers within the TTL re-use the same df. A successful CAS
  commit invalidates the cache entry so the next read pulls the freshly-
  committed state. The cache is local to one process — fine for HF Spaces
  (single replica by default), but it'd need to be lifted out (Redis,
  etc.) on a multi-replica deployment.
- **Slot reservations are per-process, in-memory, 120 s TTL.** The
  validation and voting pickers reserve up to `PICKER_BATCH_SIZE = 10`
  slots per "Load next" click — they return the first to the UI and hold
  the rest. Concurrent pickers see the reservations via
  `is_slot_reserved_by_other` and walk past, avoiding CAS conflicts.
  At most one slot per row is included in a batch so a single user
  can't monopolize all three validation slots of the same prompt. Save
  handlers call `release_slot` on the just-acted slot; the rest of the
  batch stays reserved (refreshed on each subsequent "Load next" hit, so
  an actively-working annotator keeps their queue indefinitely). A user
  who navigates away with reservations held releases them when the TTL
  expires. Like the read cache, the reservation table is process-local —
  same multi-replica caveat applies.
- **`verification_mode="no_checks"` on dataset reads.** CAS commits rewrite
  the parquet shard but never `README.md`, so the dataset card's
  `num_examples` goes stale after the first save and the default
  `basic_checks` would raise `NonMatchingSplitsSizesError` on every read.
  The parquet at the pinned `revision=sha` is authoritative; file-level
  integrity is still checked by `hf_hub_download`, schema by pyarrow, and
  CAS parent-SHA matching is still enforced server-side. The visible cost
  is that the dataset's page on huggingface.co shows a stale row count;
  re-running `import_participants_info.py --push` (or any `push_to_hub`)
  refreshes the card.
- **`seed_datasets.py` is destructive.** It calls `push_to_hub` with a fresh
  Dataset, overwriting whatever was there. Don't run it casually once real
  prompts are flowing in.
- **Schema migrations.** Any change to `PROMPTS_FEATURES` or
  `PARTICIPANTS_FEATURES` in `data.py` must be matched by a re-seed (or a
  one-off migration) — the push helpers enforce `features=…` so old-shape
  rows will fail to push. Before deploying a schema change to the
  participants table, run `import_participants_info.py --push` to refresh
  the live dataset with the new column (it preserves `test_score` per
  username on re-import).
- **OAuth requires the Space.** `gr.LoginButton` only does a real OAuth
  exchange when the app is hosted on a Hugging Face Space with
  `hf_oauth: true`. Locally, Gradio mocks the profile from the
  `huggingface-cli` token, so a "participant identity" in production is
  whoever HF says they are; locally it's whoever you logged in as.
- **Self-validation guard relies on `username` matching.** It has no
  case-folding or alias resolution — HF usernames are case-sensitive in
  practice.
- **Empty-slot semantics.** `{choice: "", username: ""}` is the sentinel for
  "no one filled this yet". Don't use truthiness on `choice` alone — always
  check `username` for slot occupancy, and check
  `choice in ACCEPT_VALIDATION_CHOICES` (not `bool(choice)`) when deciding
  whether a validation slot counts as positive.
- **Leaderboard reads the whole dataset on every tab open.** The lazy-load
  on `tab.select` keeps it off page-load, but each open of the Leaderboard
  tab still pulls the full `cultural_preferences` table. Fine for a few
  thousand prompts; plan to cache or paginate if the dataset grows.
- **`gr.BarPlot` needs an initial value matching its `x`/`y`/`color`
  columns.** Constructing a BarPlot with `x="count"`/`y="metric"` but no
  `value` produces a Vega-Lite spec referring to non-existent columns; the
  client renderer throws and the whole page freezes (initial render only —
  server-side handlers still work). Pass an empty-but-correctly-shaped
  DataFrame as `value` at construction; `refresh_leaderboard` replaces it on
  tab open.
- **Models that produced `answer_a`/`answer_b` are deliberately hidden from
  the voting view** (blind A/B). They stay in the dataset for offline
  analysis.
- **Importer country map is allowlist-only.** `import_participants_info.py`'s
  `COUNTRY_PATTERNS` only covers countries seen in registrations so far; an
  unlisted country (e.g. US, UK) imports with a blank `country`. The script
  prints a warning listing unmatched non-empty values — add a pattern and
  re-run to fix one.

## Future steps

In rough priority order:

1. **Answer generation pipeline.** Today `answer_a`/`answer_b` and
   `model_{a,b}` must be populated out-of-band. Add a job that, when a prompt
   reaches three positive validations, picks a model pair and fills in the
   answers. Until that exists, the voting tab is empty for any prompt the
   organisers haven't manually answered.
2. **Pagination / caching for leaderboard.** The 30 s read cache covers
   most back-to-back tab opens, but a large dataset still pays an
   occasional cache-miss cost (full parquet download + pandas conversion).
   Either push aggregations into a separate small dataset that the Space
   refreshes on a schedule, or cache `ranking_df` / `country_counts`
   outputs (not just the raw df) since those are what the leaderboard
   actually renders.
3. **Per-(language, country) quotas.** A dashboard for organisers showing
   gaps to fill, plus enforcement on `Save prompt` so the dataset stays
   balanced.
4. **Keep guideline translations in sync.** `guidelines/guidelines_es.md` is
   the source of truth for the four-dimension framing
   ([AlKhamissi et al., 2025](https://arxiv.org/abs/2510.05931): knowledge /
   preference / dynamics / bias-trap), the seven-tag validation flow and
   per-dimension voting criteria. `guidelines_en.md` and `guidelines_pt.md`
   mirror it — when you change ES, propagate to the other two. Code blocks
   in the EN file deliberately stay in Spanish (participants write Spanish
   prompts even when the UI is in English); PT examples are Brazilian
   Portuguese with European Portuguese touches in the §1 vocabulary table.
   The bias-trap label is localized as "Trampa de sesgo" (ES), "Bias probe"
   (EN), "Armadilha de viés" (PT) — keep the trap metaphor consistent
   across all three.
5. **Better empty-state UX.** When a tab has nothing to show (e.g. no more
   prompts to validate, leaderboard before login), the message is plain
   markdown — could be a friendlier empty state with a CTA pointing to the
   next thing to do.
6. **End-to-end UI test mode.** Grading is unit-tested in
   `tests/test_grading.py`; what's still missing is a UI walkthrough that
   mocks the HF dataset and stubs OAuth so contributors can drive `app.py`
   end-to-end without an HF token.
7. **Audit log.** Optional column tracking _who changed what when_ for each
   slot, so disputed validations/votes can be traced.

### Entry-test follow-ups (known, deferred)

Small issues we know about and have explicitly chosen to ship with. None
block the hackathon; revisit when there's time.

- **UI-only gate.** Writing / Validation / Voting are hidden via
  `gr.Tab(visible=False)`, but the handlers stay wired and the write
  helpers (`save_prompt`, validation save, vote handlers) don't check
  `best_test_score`. A user who knows the Gradio API can submit anyway.
  Intentional for a trusted-audience hackathon; if abuse shows up, add a
  server-side `has_passed_test` check to those handlers.
- **95%-of-20 cliff.** `data/test-2026.json` has 22 questions but only 20
  are `classification` (the 2 `multiple_choice` ones are silently dropped
  in `load_questions`). With `TEST_PASS_THRESHOLD = 0.95`, the effective
  pass mark is **19/20** — missing a single question fails. Either lower
  the threshold (e.g. 0.9 → 18/20), grow the question bank past 20
  classifications, or render the multiple-choice questions properly.
- **EN/PT participants get the Spanish question bank.** `load_questions`
  ignores its `lang` argument and always reads `test-2026.json` (Spanish
  prompts). UI strings around the test are translated, but the question
  prompts themselves are ES. Add `test-2026-en.json` / `test-2026-pt.json`
  and key the lookup off `lang`.
- **Robustness in `parse_test_score` / `record_test_attempt`.**
  `parse_test_score` wraps `json.loads` in try/except but `float(v)` in
  the comprehension isn't guarded; `record_test_attempt` does `int(k)`
  on dict keys without guarding either. We control all writes today so
  these can't fire in practice — but a corrupted cell would crash a page
  load instead of degrading. Cheap to harden when convenient.

## Style and conventions

- Python: type hints on public functions, no docstrings on trivial wrappers,
  one-line comments only when WHY is non-obvious.
- File layout: `app.py` is presentation, `data.py` is data. Don't put HF I/O
  in `app.py`, don't put Gradio in `data.py`.
- Handler naming: `fetch_*` for read-only handlers, `save_*` for writes.
  Vote handlers are factory-built via `_vote_handler(choice)` so each one
  keeps its `gr.OAuthProfile` annotation.
- Goals/constants live at module top (`LEADERBOARD_GOAL`, `VOTE_CHOICES`).
- Don't introduce new top-level files for one-off scripts; prefer extending
  `seed_datasets.py` or adding to `data.py`.
