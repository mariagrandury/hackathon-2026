# CLAUDE.md

Guidance for AI assistants working on this repository.

## What this is

A private Hugging Face Space (Gradio SDK) for the 2026 hackathon on cultural
preferences in LLMs. Participants write prompts that probe culturally-grounded
behaviour, validate each other's prompts, and vote on the better of two model
answers. All state lives in two private datasets on the Hub.

### Architecture

- **`app.py`** — Gradio Blocks UI. Five tabs: Annotation Guidelines, Prompt
  Writing, Prompt Validation, Answer Voting, Leaderboard. Auth uses HF OAuth
  (`hf_oauth: true` in `README.md`); the logged-in HF username is the canonical
  identity used for every write. The Leaderboard tab is lazy: its
  `cultural_preferences` read + plot rebuild only fires on `tab.select`, not
  on `demo.load`, so page-loads don't pay the leaderboard cost.
- **`data.py`** — pure data layer. Schemas (`PARTICIPANTS_FEATURES`,
  `PROMPTS_FEATURES`), I/O (`load_/push_prompts_df`, `participant_info`), row
  predicates (`is_fully_validated`, `has_answers`), and aggregations for the
  leaderboard (`user_stats`, `country_counts`, `ranking_df`).
- **`seed_datasets.py`** — one-shot script that overwrites both private
  datasets with dummy rows in the current schema.
- **`guidelines.md`** — placeholder annotation guidelines, rendered as the
  first tab.
- **`requirements.txt`** — pins `gradio[oauth]==4.44.1`,
  `huggingface_hub<1.0` (Gradio 4.44 still imports the removed `HfFolder`),
  `datasets<4.0`, `pandas`. Leaderboard plots use Gradio's native `BarPlot`,
  not matplotlib.

### Datasets

Both private, owned by `mariagrandury`. Schema is the source of truth in
`data.py`; `seed_datasets.py` only re-exports it.

| Dataset                                | Columns                                                                                                                                                                                                              |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mariagrandury/hackathon_participants` | `username`, `language`, `country`, `gmail`                                                                                                                                                                           |
| `mariagrandury/cultural_preferences`   | `username`, `language`, `country`, `prompt`, `prompt_validation_{1,2,3}`, `answer_a`, `model_a`, `answer_b`, `model_b`, `answer_chosen_{1,2,3}`                                                                       |

`prompt_validation_i` is `{validated: bool, username: str}`. `answer_chosen_i`
is `{choice: str, username: str}` where `choice ∈ {"a", "b", "both", "none"}`.

Convention for the `_i` slot columns: an empty `username` means "slot not
filled". So `{validated: False, username: ""}` is "no one has validated this
slot yet", whereas `{validated: False, username: "alice-cl"}` is "alice
explicitly disagreed".

## Behavioural rules

These are user-facing rules baked into the handlers — preserve them when
refactoring:

- A user **cannot validate their own prompt**, but **can vote on their own
  prompt** once it's fully validated.
- Validation slot picker skips prompts the user authored or already validated
  in any of the three slots.
- Voting slot picker skips prompts that are not fully validated (all three
  `prompt_validation_i.validated == True`), have no `answer_a`/`answer_b`, or
  the user already voted in any of the three slots.
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

Locally, Gradio's `LoginButton` mocks the OAuth flow with whatever HF account
your CLI / `HF_TOKEN` is logged in as, so `Save prompt` only succeeds if your
username is in the participants seed (the default seed includes
`mariagrandury`).

## Deploying to Hugging Face Spaces

1. Create a private Gradio Space named `mariagrandury/hackathon-2026`.
2. In the Space settings, add an `HF_TOKEN` secret with read+write access to
   the two private datasets.
3. Push this branch to the Space repo (the `README.md` front matter is the
   Space config).

The Space auto-detects the SDK and `app_file: app.py`. OAuth Just Works
inside the Space because `hf_oauth: true` is set.

## Caveats

- **Race conditions on writes.** Validation and voting both do `load_dataset`
  → mutate one cell → `push_to_hub`. Two concurrent writers can pick the same
  slot and the second push wins. Acceptable for the hackathon, but it is the
  next thing to fix if the Space gets contended.
- **`seed_datasets.py` is destructive.** It calls `push_to_hub` with a fresh
  Dataset, overwriting whatever was there. Don't run it casually once real
  prompts are flowing in.
- **Schema migrations.** Any change to `PROMPTS_FEATURES` in `data.py` must be
  matched by a re-seed (or a one-off migration). `push_prompts_df` enforces
  `features=PROMPTS_FEATURES` so old-shape rows will fail to push.
- **OAuth requires the Space.** `gr.LoginButton` only does a real OAuth
  exchange when the app is hosted on a Hugging Face Space with
  `hf_oauth: true`. Locally, Gradio mocks the profile from the
  `huggingface-cli` token, so a "participant identity" in production is
  whoever HF says they are; locally it's whoever you logged in as.
- **Self-validation guard relies on `username` matching.** It has no
  case-folding or alias resolution — HF usernames are case-sensitive in
  practice.
- **Empty-slot semantics.** `{validated: False, username: ""}` and
  `{choice: "", username: ""}` are sentinels for "no one filled this yet".
  Don't use truthiness on `validated`/`choice` alone — always check
  `username`.
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

## Future steps

In rough priority order:

1. **Answer generation pipeline.** Today `answer_a`/`answer_b` and
   `model_{a,b}` must be populated out-of-band. Add a job that, when a prompt
   reaches three positive validations, picks a model pair and fills in the
   answers. Until that exists, the voting tab is empty for any prompt the
   organisers haven't manually answered.
2. **Atomic row updates.** Replace the load-mutate-push pattern with
   `huggingface_hub` commit-style updates so concurrent writers don't clobber
   each other.
3. **Pagination / caching for leaderboard.** Cache `load_prompts_df()` for a
   few seconds, or push aggregations into a separate small dataset that the
   Space refreshes on a schedule.
4. **Per-(language, country) quotas.** A dashboard for organisers showing
   gaps to fill, plus enforcement on `Save prompt` so the dataset stays
   balanced.
5. **Real annotation guidelines.** Replace `guidelines.md`, possibly with a
   versioned per-language variant.
6. **Better empty-state UX.** When a tab has nothing to show (e.g. no more
   prompts to validate, leaderboard before login), the message is plain
   markdown — could be a friendlier empty state with a CTA pointing to the
   next thing to do.
7. **Self-test mode.** Vendor the in-session smoke-run pattern (mock dataset
   + stubbed OAuth) into a `tests/` folder so contributors can run the UI
   end-to-end without an HF token.
8. **Audit log.** Optional column tracking *who changed what when* for each
   slot, so disputed validations/votes can be traced.

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
