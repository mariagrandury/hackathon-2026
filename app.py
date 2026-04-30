"""Gradio Space for the 2026 hackathon.

Three tabs:
  * Annotation Guidelines — renders ``guidelines.md``.
  * Prompt Writing — appends a new row to ``cultural_preferences``.
  * Prompt Validation — fills the next free ``prompt_validation_i`` slot of
    a row written by someone else.

Authentication relies on Hugging Face OAuth (``hf_oauth: true`` in
``README.md``); the logged-in user's HF username is used as ``username``.
The Space must have an ``HF_TOKEN`` secret with read/write access to the
private datasets.
"""

from __future__ import annotations

import os
from typing import Optional

import gradio as gr
import pandas as pd
from datasets import Dataset, load_dataset

PARTICIPANTS_REPO = "mariagrandury/hackathon_participants"
PROMPTS_REPO = "mariagrandury/cultural_preferences"
GUIDELINES_PATH = "guidelines.md"

HF_TOKEN = os.environ.get("HF_TOKEN")

EMPTY_VALIDATION = {"validated": False, "username": ""}


def _read_guidelines() -> str:
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _load_participants_df() -> pd.DataFrame:
    ds = load_dataset(PARTICIPANTS_REPO, split="train", token=HF_TOKEN)
    return ds.to_pandas()


def _load_prompts_df() -> pd.DataFrame:
    ds = load_dataset(PROMPTS_REPO, split="train", token=HF_TOKEN)
    return ds.to_pandas()


def _push_prompts_df(df: pd.DataFrame) -> None:
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(
        PROMPTS_REPO, private=True, token=HF_TOKEN
    )


def _participant_info(username: str) -> Optional[dict]:
    df = _load_participants_df()
    matches = df[df["username"] == username]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def show_user(profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "Not logged in. Click **Sign in with Hugging Face** to start."
    return f"Logged in as **{profile.username}**."


def save_prompt(prompt: str, profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "Please log in with Hugging Face first."
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty."
    info = _participant_info(profile.username)
    if info is None:
        return (
            f"User `{profile.username}` is not registered as a hackathon "
            "participant. Ask the organisers to add you."
        )

    df = _load_prompts_df()
    new_row = {
        "username": profile.username,
        "language": info["language"],
        "country": info["country"],
        "prompt": prompt.strip(),
        "prompt_validation_1": dict(EMPTY_VALIDATION),
        "prompt_validation_2": dict(EMPTY_VALIDATION),
        "prompt_validation_3": dict(EMPTY_VALIDATION),
        "answer_a": "",
        "model_a": "",
        "answer_b": "",
        "model_b": "",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _push_prompts_df(df)
    return "Prompt saved. ¡Gracias!"


def fetch_next_validation(
    profile: gr.OAuthProfile | None,
) -> tuple[int, int, str, str]:
    """Return ``(row_index, slot, prompt_text, status_message)``.

    ``row_index`` and ``slot`` are ``-1`` when no prompt is available.
    """
    if profile is None:
        return -1, -1, "", "Please log in with Hugging Face first."

    df = _load_prompts_df()
    for idx, row in df.iterrows():
        if row["username"] == profile.username:
            continue
        already = any(
            row[f"prompt_validation_{i}"]["username"] == profile.username
            for i in (1, 2, 3)
        )
        if already:
            continue
        for i in (1, 2, 3):
            slot = row[f"prompt_validation_{i}"]
            if not slot["username"]:
                return int(idx), i, row["prompt"], (
                    f"Validating prompt #{int(idx)} (slot {i})."
                )
    return -1, -1, "", "No more prompts available for validation right now."


def save_validation(
    idx: int, slot: int, ok: bool, profile: gr.OAuthProfile | None
) -> str:
    if profile is None:
        return "Please log in with Hugging Face first."
    if idx is None or idx < 0 or slot not in (1, 2, 3):
        return "Load a prompt first."
    df = _load_prompts_df()
    if idx >= len(df):
        return "Prompt index out of range — try loading a new one."
    df.at[idx, f"prompt_validation_{slot}"] = {
        "validated": bool(ok),
        "username": profile.username,
    }
    _push_prompts_df(df)
    return "Validation saved."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hackathon 2026") as demo:
        gr.Markdown("# Hackathon 2026 — Cultural Preferences")

        with gr.Row():
            gr.LoginButton()
            user_md = gr.Markdown("Not logged in.")

        demo.load(show_user, inputs=None, outputs=user_md)

        with gr.Tabs():
            with gr.Tab("Annotation Guidelines"):
                gr.Markdown(_read_guidelines())

            with gr.Tab("Prompt Writing"):
                prompt_box = gr.Textbox(
                    label="Your prompt",
                    lines=5,
                    placeholder="Write a culturally-grounded prompt…",
                )
                save_btn = gr.Button("Save", variant="primary")
                save_status = gr.Markdown()
                save_btn.click(
                    save_prompt, inputs=prompt_box, outputs=save_status
                )

            with gr.Tab("Prompt Validation"):
                idx_state = gr.State(-1)
                slot_state = gr.State(-1)
                load_status = gr.Markdown(
                    "Click **Load next prompt** to start validating."
                )
                current_prompt = gr.Textbox(
                    label="Prompt to validate",
                    lines=5,
                    interactive=False,
                )
                ok_check = gr.Checkbox(label="OK", value=False)
                with gr.Row():
                    load_btn = gr.Button("Load next prompt")
                    save_val_btn = gr.Button("Save validation", variant="primary")
                save_val_status = gr.Markdown()

                load_btn.click(
                    fetch_next_validation,
                    inputs=None,
                    outputs=[idx_state, slot_state, current_prompt, load_status],
                )
                save_val_btn.click(
                    save_validation,
                    inputs=[idx_state, slot_state, ok_check],
                    outputs=save_val_status,
                )

    return demo


if __name__ == "__main__":
    build_demo().launch()
