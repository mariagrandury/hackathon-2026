---
name: annotate-prompts
description: Have Claude annotate cultural_preferences prompts the same way human annotators do — assign each prompt a validation bucket, a blind A/B vote, the region/city it mentions, and a cultural topic — writing ONLY to the separate cultural_preferences_claude dataset. Billed against the Claude subscription (the judgment is done by Claude Code, not the Anthropic API). Use when asked to run the Claude annotator / vote as Claude / refresh Claude's labels.
---

# Annotate prompts as Claude

You (Claude Code) play the role of a human annotator: read each prompt, assign a
**validation bucket**, cast a **blind A/B vote** (for prompts with an answer
pair), and extract two descriptive fields — the **region/city** mentioned and a
**cultural topic**. The Python script `annotate_claude.py` handles all HF I/O;
**you** supply the judgment, so this runs on the Claude subscription, not the API.

## ⛔ Never touch the human datasets

Write ONLY to `mariagrandury/cultural_preferences_claude` (via
`annotate_claude.py --write`). **Never** push, commit, or `create_commit` to the
human datasets — `mariagrandury/cultural_preferences` or
`mariagrandury/hackathon_participants`. Claude is a measured baseline, not a
participant: its labels live in their own dataset and must never fill a human
validation/vote slot, alter the human consensus, or touch the voting gate. The
only supported write path is `annotate_claude.write_claude_annotations`, which is
hard-wired to the Claude repo — do not write Hub data any other way.

## Use the SAME guidelines as the humans

The judgment criteria are the participant guidelines, not a private rubric.
**Read `guidelines/guidelines_en.md` §1.5 (four dimensions), §3 (validation), and
§4 (voting) before labelling** — that is the exact same guidance humans follow
(EN is a faithful translation of the Spanish source). The prompts themselves are
in Spanish/Portuguese; judge them against those guidelines.

If you find any criterion ambiguous or improvable while labelling, do **not**
silently invent a private clarification here — propose the wording change to the
human guidelines (`guidelines_es.md` is the source of truth; mirror to
`guidelines_en.md` and `guidelines_pt.md`) so humans and Claude stay aligned.

## Procedure

Run from the repo root with the venv active (`source .venv/bin/activate`).

### 1. Dump the work-list

```bash
python annotate_claude.py --dump-pending --out /tmp/annotate_pending.json [--limit N] [--tasks both|validation|voting]
```

Each item: `{id, country, language, prompt, needs_validation, needs_vote}` plus
`answer_a`/`answer_b` when `needs_vote` is true (model names withheld — votes are
blind). Use `--limit` for a trial batch; omit it for the full backlog.

### 2. Judge each prompt (per `guidelines_en.md`)

For every item produce:

- **`validation_choice`** — exactly one of seven buckets. Reject: `trivial`,
  `stereotype`, `unrelated`. Accept (the four cultural dimensions, pick the
  _predominant_ one): `knowledge`, `preference`, `dynamics`, `bias_probe`.
  (See §3, incl. the trivial-vs-knowledge and stereotype-vs-bias_probe tables.)
- **`vote_choice`** — only when `needs_vote` is true: `a` / `b` (one clearly
  better for the prompt's culture and role) / `both` (both correct and natural) /
  `none` (both seriously wrong). Judge from the role in the prompt; reward the
  locally-natural language variant, not "neutral" grammar. (See §4.)
- **`region`** — the city / region / comarca the prompt names, verbatim and as
  specific as stated (e.g. `Os Ancares`, `ría de Arousa`, `Picos de Europa`,
  `Santiago de Compostela`). Empty string if the prompt names no place below the
  country level.
- **`cultural_topic`** — exactly one key from the taxonomy below (predominant
  facet). Use `other` only if nothing fits.

Give a short `validation_reason` / `vote_reason` (one sentence each, in English).

#### Cultural-topic taxonomy

Grounded in the benchmarks AlKhamissi et al. (2025, arXiv:2510.05931) surveys
(BLEnD, CANDLE, CultureAtlas) — that paper defines the _dimensions_, not topics.

| key                          | what it covers                                                                |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `food_and_drink`             | cuisine, dishes, ingredients, beverages, meals, eating habits                 |
| `festivals_and_celebrations` | festivals, holidays, religious/civic celebrations, rites of passage           |
| `religion_and_beliefs`       | religious practice, spirituality, folk beliefs, the sacred                    |
| `family_and_kinship`         | family roles, household structure, kinship, generational/gender roles at home |
| `social_norms_and_etiquette` | politeness, greetings, gift-giving, hospitality, taboos                       |
| `work_and_occupations`       | trades, livelihoods, rural/artisanal jobs, daily working life, economy        |
| `language_and_dialect`       | local language, dialect, slang, accent, sayings, oral expression              |
| `arts_music_dance`           | music, dance, visual/performing arts, crafts, folklore performance            |
| `clothing_and_appearance`    | traditional dress, garments, adornment, grooming                              |
| `traditions_and_customs`     | inherited customs/rituals/folk practices not tied to one festival             |
| `geography_and_local_life`   | place, landscape, climate, regional identity, rural/urban daily life          |
| `history_and_heritage`       | historical events, collective memory, heritage sites, ancestry                |
| `values_and_opinions`        | attitudes, regional opinions, identity, in-group/out-group views              |
| `sports_and_leisure`         | sports, games, pastimes, recreation, fan culture                              |
| `other`                      | culturally-grounded but fits none of the above                                |

### 3. Scale the labelling to the batch size

- **Small batch (≲40 prompts):** judge inline and write the JSON yourself.
- **Large backlog (hundreds+):** fan out with the **Agent** tool — split into
  chunks of ~25, one subagent per chunk (subagents bill on the same
  subscription). Give each subagent its chunk **and** `guidelines_en.md`
  §1.5/§3/§4 + the topic taxonomy, and have it **return a JSON array** of
  `{id, validation_choice, validation_reason, vote_choice, vote_reason, region, cultural_topic}`
  (omit vote fields when the item had no answer pair). Concatenate the chunks.
  If the user explicitly opted into a Workflow ("use a workflow" / "ultracode"),
  a `pipeline`/`parallel` fan-out with a structured-output schema is the cleaner
  instrument for the full backlog.

### 4. Write the results

One JSON array at `/tmp/annotate_results.json`, each element
`{id, validation_choice, validation_reason, vote_choice, vote_reason, region, cultural_topic}`
(omit vote fields for non-votable prompts), then:

```bash
python annotate_claude.py --write /tmp/annotate_results.json \
    --source trial|full|incremental --calls N [--tokens-in N --tokens-out N]
```

The script validates every `validation_choice` / `vote_choice` / `cultural_topic`
against the allowed sets before pushing, stamps the model + UTC timestamp, and
**upserts** by `id` (a later voting pass merges into the row a validation pass
created, preserving region/topic).

**Always record usage.** Every `--write` appends a line to `annotate_runs.jsonl`
(prompts, calls, tokens) so we can extrapolate cost as more prompts arrive:

- `--calls N`: number of subagents/labelling calls (1 if you judged inline).
- `--tokens-in` / `--tokens-out`: **measured** tokens when you have them (e.g.
  from a Workflow's `budget.spent()`); omit and the script falls back to a
  payload-size **estimate** (flagged as such in the ledger).
- `--source`: `trial` (capped batch), `full` (backlog), `incremental` (catch-up).

### 5. Report

```bash
python annotate_claude.py --report   # coverage, buckets, topics, agreement
python annotate_claude.py --usage    # cumulative cost + extrapolation
```

## Optional: analysis.md dimension pass (D1–D6)

A second, independent annotation layer that classifies every prompt across the
six dimensions defined in `guidelines/analysis.md` (D1 cultural dimension, D2
thematic taxonomy, D3 register, D4 linguistic complexity, D5 multilingual level,
and D6 cultural-anchoring exposed as `d6_anchoring`). These land as the
`d1_dimension … d6_anchoring` columns and **merge into the existing row per `id`**
(non-destructive — validation/vote/region/topic are preserved).

1. `python annotate_claude.py --dump-dims --out /tmp/annotate_dims.json [--limit N]`
   — every prompt with no `d1_dimension` yet. Each item keeps `system_prompt` and
   `prompt` **separate** (the rubric labels the prompt but reads the system prompt
   for role context).
2. Read `guidelines/analysis.md` (the full rubric, in Spanish, with the decision
   order and the tie-break tables) and classify each prompt. Emit per item
   `{id, d1_dimension, d2_topic, d3_register, d4_complexity, d5_multilingual, d6_anchoring}`
   with the **exact Spanish values** from the rubric (accents/case are
   auto-normalized on write; unknown values are skipped, not fatal).
3. Fan out the same way (chunks + Agent/Workflow) for the full set.
4. `python annotate_claude.py --write /tmp/dims_results.json --source dims --calls N`.

`--report` prints the per-dimension distributions once written.

## Notes

- The `cultural_preferences_claude` repo is created automatically on first write.
- Re-running is safe and incremental: `--dump-pending` only surfaces prompts
  Claude hasn't already labelled, so run this repeatedly to pick up new prompts.
