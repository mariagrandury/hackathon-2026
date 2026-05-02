"""Gradio Space for the 2026 hackathon.

Tabs:
  * Annotation Guidelines — renders ``guidelines.md``.
  * Prompt Writing — appends a new row to ``cultural_preferences``.
  * Prompt Validation — fills the next free ``prompt_validation_i`` slot of
    a row written by someone else.
  * Answer Voting — for fully-validated prompts, lets the user vote on the
    better of two model answers (or flag both/none) into ``answer_chosen_i``.

Authentication relies on Hugging Face OAuth (``hf_oauth: true`` in
``README.md``); the logged-in user's HF username is used as ``username``.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from data import (
    EMPTY_VALIDATION,
    EMPTY_VOTE,
    has_answers,
    is_fully_validated,
    load_prompts_df,
    participant_info,
    push_prompts_df,
)

GUIDELINES_PATH = "guidelines.md"
VOTE_CHOICES = ("a", "b", "both", "none")


def _read_guidelines() -> str:
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


def show_user(profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "Not logged in. Click **Sign in with Hugging Face** to start."
    return f"Logged in as **{profile.username}**."


# ---------------------------------------------------------------------------
# Prompt writing
# ---------------------------------------------------------------------------


def save_prompt(prompt: str, profile: gr.OAuthProfile | None) -> str:
    if profile is None:
        return "Please log in with Hugging Face first."
    if not prompt or not prompt.strip():
        return "Prompt cannot be empty."
    info = participant_info(profile.username)
    if info is None:
        return (
            f"User `{profile.username}` is not registered as a hackathon "
            "participant. Ask the organisers to add you."
        )

    df = load_prompts_df()
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
        "answer_chosen_1": dict(EMPTY_VOTE),
        "answer_chosen_2": dict(EMPTY_VOTE),
        "answer_chosen_3": dict(EMPTY_VOTE),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    push_prompts_df(df)
    return "Prompt saved. ¡Gracias!"


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------


def fetch_next_validation(profile: gr.OAuthProfile | None):
    """Return ``(idx, slot, prompt_text, status)`` for the next prompt the
    user can validate. ``idx == -1`` means nothing to do."""
    if profile is None:
        return -1, -1, "", "Please log in with Hugging Face first."
    df = load_prompts_df()
    for idx, row in df.iterrows():
        if row["username"] == profile.username:
            continue
        if any(
            row[f"prompt_validation_{i}"]["username"] == profile.username
            for i in (1, 2, 3)
        ):
            continue
        for i in (1, 2, 3):
            if not row[f"prompt_validation_{i}"]["username"]:
                return (
                    int(idx),
                    i,
                    row["prompt"],
                    f"Validating prompt #{int(idx)} (slot {i}).",
                )
    return -1, -1, "", "No more prompts available for validation right now."


def save_validation(
    idx: int, slot: int, ok: bool, profile: gr.OAuthProfile | None
) -> str:
    if profile is None:
        return "Please log in with Hugging Face first."
    if idx is None or idx < 0 or slot not in (1, 2, 3):
        return "Load a prompt first."
    df = load_prompts_df()
    if idx >= len(df):
        return "Prompt index out of range — try loading a new one."
    df.at[idx, f"prompt_validation_{slot}"] = {
        "validated": bool(ok),
        "username": profile.username,
    }
    push_prompts_df(df)
    return "Validation saved."


# ---------------------------------------------------------------------------
# Answer voting
# ---------------------------------------------------------------------------


def fetch_next_voting(profile: gr.OAuthProfile | None):
    """Return ``(idx, slot, prompt, answer_a, answer_b, status)`` for the next
    fully-validated prompt the user can vote on. ``idx == -1`` means nothing.

    Users *can* vote on their own prompts (unlike validation).
    """
    if profile is None:
        return -1, -1, "", "", "", "Please log in with Hugging Face first."
    df = load_prompts_df()
    for idx, row in df.iterrows():
        if not is_fully_validated(row) or not has_answers(row):
            continue
        if any(
            row[f"answer_chosen_{i}"]["username"] == profile.username
            for i in (1, 2, 3)
        ):
            continue
        for i in (1, 2, 3):
            if not row[f"answer_chosen_{i}"]["username"]:
                return (
                    int(idx),
                    i,
                    row["prompt"],
                    row["answer_a"],
                    row["answer_b"],
                    f"Voting on prompt #{int(idx)} (slot {i}).",
                )
    return (
        -1,
        -1,
        "",
        "",
        "",
        "No more validated prompts available for voting right now.",
    )


def save_vote(
    idx: int, slot: int, choice: str, profile: gr.OAuthProfile | None
):
    """Persist the vote and return the next prompt to vote on."""
    if profile is None:
        return -1, -1, "", "", "", "Please log in with Hugging Face first."
    if (
        idx is None
        or idx < 0
        or slot not in (1, 2, 3)
        or choice not in VOTE_CHOICES
    ):
        return -1, -1, "", "", "", "Load a prompt first."
    df = load_prompts_df()
    if idx >= len(df):
        return (
            -1,
            -1,
            "",
            "",
            "",
            "Prompt index out of range — try loading a new one.",
        )
    df.at[idx, f"answer_chosen_{slot}"] = {
        "choice": choice,
        "username": profile.username,
    }
    push_prompts_df(df)
    return fetch_next_voting(profile)


def _vote_handler(choice: str):
    """Build a Gradio handler that records ``choice`` and loads the next prompt.

    A factory is needed so that each click handler keeps its own ``OAuthProfile``
    annotation — Gradio only auto-injects the profile when it sees the
    annotation on the function signature."""

    def handler(idx: int, slot: int, profile: gr.OAuthProfile | None):
        return save_vote(idx, slot, choice, profile)

    return handler


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _build_writing_tab() -> None:
    prompt_box = gr.Textbox(
        label="Your prompt",
        lines=5,
        placeholder="Write a culturally-grounded prompt…",
    )
    save_btn = gr.Button("Save", variant="primary")
    save_status = gr.Markdown()
    save_btn.click(save_prompt, inputs=prompt_box, outputs=save_status)


def _build_validation_tab() -> None:
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


def _build_voting_tab() -> None:
    idx_state = gr.State(-1)
    slot_state = gr.State(-1)
    status_md = gr.Markdown("Click **Load next prompt** to start voting.")
    current_prompt = gr.Textbox(label="Prompt", lines=4, interactive=False)
    with gr.Row():
        ans_a = gr.Textbox(label="Answer A", lines=8, interactive=False)
        ans_b = gr.Textbox(label="Answer B", lines=8, interactive=False)
    with gr.Row():
        load_btn = gr.Button("Load next prompt")
        a_btn = gr.Button("A is better", variant="primary")
        b_btn = gr.Button("B is better", variant="primary")
        both_btn = gr.Button("Both good")
        none_btn = gr.Button("No good")

    fetch_outputs = [
        idx_state,
        slot_state,
        current_prompt,
        ans_a,
        ans_b,
        status_md,
    ]
    load_btn.click(fetch_next_voting, inputs=None, outputs=fetch_outputs)
    for btn, choice in (
        (a_btn, "a"),
        (b_btn, "b"),
        (both_btn, "both"),
        (none_btn, "none"),
    ):
        btn.click(
            _vote_handler(choice),
            inputs=[idx_state, slot_state],
            outputs=fetch_outputs,
        )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Hackathon 2026") as demo:
        gr.Markdown("# Hackathon 2026 — Cultural Preferences")

        with gr.Row():
            gr.LoginButton()
            user_md = gr.Markdown("Not logged in.")

        with gr.Tabs():
            with gr.Tab("Annotation Guidelines"):
                gr.Markdown(_read_guidelines())
            with gr.Tab("Prompt Writing"):
                _build_writing_tab()
            with gr.Tab("Prompt Validation"):
                _build_validation_tab()
            with gr.Tab("Answer Voting"):
                _build_voting_tab()

        demo.load(show_user, inputs=None, outputs=user_md)
    return demo


if __name__ == "__main__":
    build_demo().launch()
