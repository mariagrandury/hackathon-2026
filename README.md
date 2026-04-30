---
title: Hackathon 2026
emoji: 🌎
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - read-repos
---

# Hackathon 2026 — Cultural Preferences

Gradio Space used during the hackathon to let participants:

1. Read the annotation guidelines.
2. Write prompts that probe cultural preferences.
3. Validate prompts written by other participants.

## Datasets

The Space reads from and writes to two private Hugging Face datasets:

- `mariagrandury/hackathon_participants` — registered participants (read-only).
- `mariagrandury/cultural_preferences` — prompts and validations (read/write).

Access requires the `HF_TOKEN` Space secret with read/write permissions on both
private repos.

## Local development

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...      # token with access to the private datasets
python seed_datasets.py     # one-shot: create the dummy datasets
python app.py
```

## Deploying to Hugging Face Spaces

1. Create a new (private) Space named `mariagrandury/hackathon-2026` with
   the Gradio SDK.
2. Add `HF_TOKEN` as a Space secret.
3. Push this repository to the Space.
