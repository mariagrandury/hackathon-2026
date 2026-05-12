"""Gradio Space for the 2026 hackathon.

Tabs:
  * Annotation Guidelines — renders ``guidelines/guidelines_{lang}.md``.
  * Prompt Writing — appends a new row to ``cultural_preferences``.
  * Prompt Validation — fills the next free ``prompt_validation_i`` slot of
    a row written by someone else.
  * Answer Voting — for fully-validated prompts, lets the user vote on the
    better of two model answers (or flag both/none) into ``answer_chosen_i``.
  * Leaderboard — per-user progress (against a goal), full ranking, and a
    per-country split of validated vs pending prompts.

Authentication relies on Hugging Face OAuth (``hf_oauth: true`` in
``README.md``); the logged-in user's HF username is used as ``username``.

The whole interface is available in English, Spanish and Portuguese. The
language is read from the participant's ``language`` field in the
``hackathon_participants`` dataset on every page load (i.e. after the OAuth
redirect), so participants always see the UI in the language they signed up
with — no manual toggle. Logged-out visitors see the default (English).
Handlers that produce dynamic status messages take the resolved language as
a regular input and look up the right string in ``T``.
"""

from __future__ import annotations

import logging
import os

import gradio as gr
import pandas as pd

log = logging.getLogger("hackathon")

from data import (
    EMPTY_VALIDATION,
    EMPTY_VOTE,
    country_counts,
    country_display,
    has_answers,
    is_fully_validated,
    load_prompts_df,
    participant_info,
    push_prompts_df,
    ranking_df,
    user_stats,
)

VOTE_CHOICES = ("a", "b", "both", "none")
VALIDATION_CHOICES = ("trivial", "stereotype", "unrelated", "relevant")
LEADERBOARD_GOAL = 100
DEFAULT_LANG = "en"
GUIDELINES_DIR = "guidelines"
IMAGES_DIR = "images"

# Register ``images/`` as a static directory so the guidelines Markdown can
# embed local infographics via ``/file=images/...`` URLs. Without this,
# Gradio's ``/file=`` route refuses the request and the browser only
# renders the alt text.
gr.set_static_paths([IMAGES_DIR])


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

T: dict[str, dict[str, str]] = {
    "en": {
        "title": "# Hackathon 2026 — Cultural Preferences",
        "login_button": "Sign in with Hugging Face",
        "not_logged_in": "Not logged in. Click **Sign in with Hugging Face** to start.",
        "not_logged_in_short": "Not logged in.",
        "logged_in_as": "Logged in as **{username}**.",
        "tab_guidelines": "Annotation Guidelines",
        "tab_writing": "Prompt Writing",
        "tab_validation": "Prompt Validation",
        "tab_voting": "Answer Voting",
        "tab_leaderboard": "Leaderboard",
        "guidelines_missing": "Guidelines for this language are not available yet.",
        # Common
        "login_required": "Please log in with Hugging Face first.",
        "load_first": "Load a prompt first.",
        "out_of_range": "Prompt index out of range — try loading a new one.",
        # Writing
        "writing_system_label": "System prompt (optional)",
        "writing_system_placeholder": 'Steering instructions for the model, e.g. "Respond in Spanish without inventing data."',
        "writing_prompt_label": "Your prompt",
        "writing_prompt_placeholder": "Write a culturally-grounded prompt…",
        "writing_save_button": "Save",
        "writing_empty": "Prompt cannot be empty.",
        "writing_not_participant": "User `{username}` is not registered as a hackathon participant. Ask the organisers to add you.",
        "writing_saved": "Prompt saved. Thanks!",
        # Merged display (validation + voting tabs)
        "merged_system_header": "System prompt",
        "merged_prompt_header": "Prompt",
        # Validation
        "validation_load_status_initial": "Click **Load next prompt** to start validating.",
        "validation_prompt_label": "Prompt to validate",
        "validation_choice_label": "How would you classify this prompt?",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Stereotyping / non-neutral",
        "validation_choice_unrelated": "Unrelated to any country's culture",
        "validation_choice_relevant": "Relevant for understanding a country's culture",
        "validation_choice_required": "Pick one of the four options before saving.",
        "validation_load_button": "Load next prompt",
        "validation_save_button": "Save validation",
        "validation_in_progress": "Validating prompt #{id} ({country}).",
        "validation_no_more": "No more prompts available for validation right now.",
        "validation_saved": "Validation saved.",
        # Voting
        "voting_load_status_initial": "Click **Load next prompt** to start voting.",
        "voting_prompt_label": "Prompt",
        "voting_answer_a_label": "Answer A",
        "voting_answer_b_label": "Answer B",
        "voting_load_button": "Load next prompt",
        "voting_a_button": "A is better",
        "voting_b_button": "B is better",
        "voting_both_button": "Both good",
        "voting_none_button": "No good",
        "voting_in_progress": "Voting on prompt #{id} ({country}).",
        "voting_no_more": "No more validated prompts available for voting right now.",
        # Leaderboard
        "leaderboard_refresh_button": "Refresh",
        "leaderboard_user_plot_label": "Your progress (goal = {goal})",
        "leaderboard_ranking_label": "Ranking — by prompts sent",
        "leaderboard_country_plot_label": "Prompts by country: validated (green) vs pending (yellow)",
        # Plot internals
        "plot_metric_sent": "Prompts sent",
        "plot_metric_validated": "Prompts validated",
        "plot_metric_voted": "Answers voted",
        "plot_goal_legend": "Goal: {goal}",
        "plot_xlabel_count": "Count",
        "plot_country_no_prompts": "No prompts yet",
        "plot_country_fully": "Fully validated",
        "plot_country_pending": "Pending validation",
        "plot_country_xlabel": "Country",
        "plot_country_ylabel": "Prompts",
        # Ranking columns
        "ranking_col_username": "username",
        "ranking_col_sent": "prompts sent",
        "ranking_col_validated": "prompts validated",
        "ranking_col_voted": "answers voted",
    },
    "es": {
        "title": "# Hackathon 2026 — Preferencias Culturales",
        "login_button": "Iniciar sesión con Hugging Face",
        "not_logged_in": "Sesión no iniciada. Haz clic en **Iniciar sesión con Hugging Face** para empezar.",
        "not_logged_in_short": "Sesión no iniciada.",
        "logged_in_as": "Sesión iniciada como **{username}**.",
        "tab_guidelines": "Pautas de anotación",
        "tab_writing": "Escribir prompts",
        "tab_validation": "Validar prompts",
        "tab_voting": "Votar respuestas",
        "tab_leaderboard": "Ranking",
        "guidelines_missing": "Las pautas en este idioma todavía no están disponibles.",
        # Common
        "login_required": "Por favor, inicia sesión con Hugging Face primero.",
        "load_first": "Carga un prompt primero.",
        "out_of_range": "Índice de prompt fuera de rango — intenta cargar uno nuevo.",
        # Writing
        "writing_system_label": "System prompt",
        "writing_system_placeholder": 'Instrucciones para el modelo, p. ej. "Responde en español sin inventar datos."',
        "writing_prompt_label": "Tu prompt",
        "writing_prompt_placeholder": "Escribe un prompt con base cultural…",
        "writing_save_button": "Guardar",
        "writing_empty": "El prompt no puede estar vacío.",
        "writing_not_participant": "El usuario `{username}` no está registrado como participante del hackathon. Pide a los organizadores que te añadan.",
        "writing_saved": "Prompt guardado. ¡Gracias!",
        # Merged display (validation + voting tabs)
        "merged_system_header": "System prompt",
        "merged_prompt_header": "Prompt",
        # Validation
        "validation_load_status_initial": "Haz clic en **Cargar siguiente prompt** para empezar a validar.",
        "validation_prompt_label": "Prompt a validar",
        "validation_choice_label": "¿Cómo clasificarías este prompt?",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Estereotipos / no neutral",
        "validation_choice_unrelated": "No relacionado con la cultura de un país",
        "validation_choice_relevant": "Relevante para comprender la cultura de un país",
        "validation_choice_required": "Selecciona una de las cuatro opciones antes de guardar.",
        "validation_load_button": "Cargar siguiente prompt",
        "validation_save_button": "Guardar validación",
        "validation_in_progress": "Validando el prompt #{id} ({country}).",
        "validation_no_more": "No hay más prompts disponibles para validar ahora mismo.",
        "validation_saved": "Validación guardada.",
        # Voting
        "voting_load_status_initial": "Haz clic en **Cargar siguiente prompt** para empezar a votar.",
        "voting_prompt_label": "Prompt",
        "voting_answer_a_label": "Respuesta A",
        "voting_answer_b_label": "Respuesta B",
        "voting_load_button": "Cargar siguiente prompt",
        "voting_a_button": "A es mejor",
        "voting_b_button": "B es mejor",
        "voting_both_button": "Ambas buenas",
        "voting_none_button": "Ninguna buena",
        "voting_in_progress": "Votando el prompt #{id} ({country}).",
        "voting_no_more": "No hay más prompts validados disponibles para votar ahora mismo.",
        # Leaderboard
        "leaderboard_refresh_button": "Actualizar",
        "leaderboard_user_plot_label": "Tu progreso (objetivo = {goal})",
        "leaderboard_ranking_label": "Clasificación — por prompts enviados",
        "leaderboard_country_plot_label": "Prompts por país: validados (verde) vs pendientes (amarillo)",
        # Plot internals
        "plot_metric_sent": "Prompts enviados",
        "plot_metric_validated": "Prompts validados",
        "plot_metric_voted": "Respuestas votadas",
        "plot_goal_legend": "Objetivo: {goal}",
        "plot_xlabel_count": "Cantidad",
        "plot_country_no_prompts": "Aún no hay prompts",
        "plot_country_fully": "Totalmente validados",
        "plot_country_pending": "Pendientes de validación",
        "plot_country_xlabel": "País",
        "plot_country_ylabel": "Prompts",
        # Ranking columns
        "ranking_col_username": "usuario",
        "ranking_col_sent": "prompts enviados",
        "ranking_col_validated": "prompts validados",
        "ranking_col_voted": "respuestas votadas",
    },
    "pt": {
        "title": "# Hackathon 2026 — Preferências Culturais",
        "login_button": "Entrar com o Hugging Face",
        "not_logged_in": "Sessão não iniciada. Clique em **Entrar com o Hugging Face** para começar.",
        "not_logged_in_short": "Sessão não iniciada.",
        "logged_in_as": "Sessão iniciada como **{username}**.",
        "tab_guidelines": "Diretrizes de anotação",
        "tab_writing": "Escrever prompts",
        "tab_validation": "Validar prompts",
        "tab_voting": "Votar respostas",
        "tab_leaderboard": "Classificação",
        "guidelines_missing": "As diretrizes neste idioma ainda não estão disponíveis.",
        # Common
        "login_required": "Por favor, entre com o Hugging Face primeiro.",
        "load_first": "Carregue um prompt primeiro.",
        "out_of_range": "Índice do prompt fora do intervalo — tente carregar um novo.",
        # Writing
        "writing_system_label": "Mensagem de sistema (opcional)",
        "writing_system_placeholder": 'Instruções para o modelo, p. ex. "Responda em português sem inventar dados."',
        "writing_prompt_label": "Seu prompt",
        "writing_prompt_placeholder": "Escreva um prompt culturalmente fundamentado…",
        "writing_save_button": "Salvar",
        "writing_empty": "O prompt não pode estar vazio.",
        "writing_not_participant": "O usuário `{username}` não está registrado como participante do hackathon. Peça aos organizadores para adicioná-lo.",
        "writing_saved": "Prompt salvo. Obrigado!",
        # Merged display (validation + voting tabs)
        "merged_system_header": "Mensagem de sistema",
        "merged_prompt_header": "Prompt",
        # Validation
        "validation_load_status_initial": "Clique em **Carregar próximo prompt** para começar a validar.",
        "validation_prompt_label": "Prompt a validar",
        "validation_choice_label": "Como você classificaria este prompt?",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Estereótipos / não neutro",
        "validation_choice_unrelated": "Não relacionado com a cultura de um país",
        "validation_choice_relevant": "Relevante para compreender a cultura de um país",
        "validation_choice_required": "Selecione uma das quatro opções antes de salvar.",
        "validation_load_button": "Carregar próximo prompt",
        "validation_save_button": "Salvar validação",
        "validation_in_progress": "Validando o prompt #{id} ({country}).",
        "validation_no_more": "Não há mais prompts disponíveis para validação no momento.",
        "validation_saved": "Validação salva.",
        # Voting
        "voting_load_status_initial": "Clique em **Carregar próximo prompt** para começar a votar.",
        "voting_prompt_label": "Prompt",
        "voting_answer_a_label": "Resposta A",
        "voting_answer_b_label": "Resposta B",
        "voting_load_button": "Carregar próximo prompt",
        "voting_a_button": "A é melhor",
        "voting_b_button": "B é melhor",
        "voting_both_button": "Ambas boas",
        "voting_none_button": "Nenhuma boa",
        "voting_in_progress": "Votando no prompt #{id} ({country}).",
        "voting_no_more": "Não há mais prompts validados disponíveis para votação no momento.",
        # Leaderboard
        "leaderboard_refresh_button": "Atualizar",
        "leaderboard_user_plot_label": "Seu progresso (meta = {goal})",
        "leaderboard_ranking_label": "Classificação — por prompts enviados",
        "leaderboard_country_plot_label": "Prompts por país: validados (verde) vs pendentes (amarelo)",
        # Plot internals
        "plot_metric_sent": "Prompts enviados",
        "plot_metric_validated": "Prompts validados",
        "plot_metric_voted": "Respostas votadas",
        "plot_goal_legend": "Meta: {goal}",
        "plot_xlabel_count": "Contagem",
        "plot_country_no_prompts": "Ainda não há prompts",
        "plot_country_fully": "Totalmente validados",
        "plot_country_pending": "Validação pendente",
        "plot_country_xlabel": "País",
        "plot_country_ylabel": "Prompts",
        # Ranking columns
        "ranking_col_username": "usuário",
        "ranking_col_sent": "prompts enviados",
        "ranking_col_validated": "prompts validados",
        "ranking_col_voted": "respostas votadas",
    },
}


def _t(lang: str | None) -> dict[str, str]:
    return T.get(lang or DEFAULT_LANG, T[DEFAULT_LANG])


def _read_guidelines(lang: str) -> str:
    path = os.path.join(GUIDELINES_DIR, f"guidelines_{lang}.md")
    if not os.path.exists(path):
        return _t(lang)["guidelines_missing"]
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------


def show_user(lang: str, profile: gr.OAuthProfile | None) -> str:
    s = _t(lang)
    if profile is None:
        return s["not_logged_in"]
    return s["logged_in_as"].format(username=profile.username)


def _merged_prompt_display(lang: str, system_prompt: str, prompt: str) -> str:
    """Format ``(system_prompt, prompt)`` for a single read-only textbox.

    Used by the validation and voting tabs (the user asked for those to be
    merged into one cell). If ``system_prompt`` is empty, the prompt is
    returned alone so we don't render a useless empty section."""
    sm = (system_prompt or "").strip()
    p = (prompt or "").strip()
    if not sm:
        return p
    s = _t(lang)
    return (
        f"[{s['merged_system_header']}]\n{sm}\n\n" f"[{s['merged_prompt_header']}]\n{p}"
    )


# ---------------------------------------------------------------------------
# Prompt writing
# ---------------------------------------------------------------------------


def save_prompt(
    system_prompt: str,
    prompt: str,
    lang: str,
    profile: gr.OAuthProfile | None,
):
    """Append a new prompt and clear the textboxes on success.

    Returns ``(status, system_box_update, prompt_box_update)``. On failure
    the textboxes are left untouched (``gr.update()`` with no args), so the
    user can fix the issue and re-submit. On success both clear, which is
    the unambiguous "your text actually went through" cue."""
    s = _t(lang)
    keep = gr.update()  # leave textbox value as-is
    if profile is None:
        return s["login_required"], keep, keep
    if not prompt or not prompt.strip():
        return s["writing_empty"], keep, keep
    info = participant_info(profile.username)
    if info is None:
        return (
            s["writing_not_participant"].format(username=profile.username),
            keep,
            keep,
        )

    df = load_prompts_df()
    next_id = int(df["id"].max()) + 1 if "id" in df.columns and len(df) > 0 else 1
    new_row = {
        "id": next_id,
        "username": profile.username,
        "language": info["language"],
        "country": info["country"],
        "system_prompt": (system_prompt or "").strip(),
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
    push_prompts_df(
        df,
        commit_message=f"{profile.username} sent prompt with ID {next_id}",
    )
    return (
        s["writing_saved"],
        gr.update(value=""),  # clear system_box
        gr.update(value=""),  # clear prompt_box
    )


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------


def fetch_next_validation(lang: str, profile: gr.OAuthProfile | None):
    """Return ``(idx, slot, prompt_text, status)`` for the next prompt the
    user can validate. ``idx == -1`` means nothing to do.

    Filters to the user's own country (only prompts grounded in their
    culture are shown). Users with no registered country fall through and
    see all prompts."""
    s = _t(lang)
    if profile is None:
        return -1, -1, "", s["login_required"]
    info = participant_info(profile.username)
    user_country = info.get("country") if info else None
    df = load_prompts_df()
    for idx, row in df.iterrows():
        if user_country and row.get("country") != user_country:
            continue
        if row["username"] == profile.username:
            continue
        if any(
            row[f"prompt_validation_{i}"]["username"] == profile.username
            for i in (1, 2, 3)
        ):
            continue
        for i in (1, 2, 3):
            if not row[f"prompt_validation_{i}"]["username"]:
                display = _merged_prompt_display(
                    lang,
                    row.get("system_prompt", ""),
                    row["prompt"],
                )
                prompt_id = int(row.get("id", idx))
                return (
                    int(idx),
                    i,
                    display,
                    s["validation_in_progress"].format(
                        id=prompt_id,
                        country=country_display(row.get("country")),
                    ),
                )
    return -1, -1, "", s["validation_no_more"]


def save_validation(
    idx: int,
    slot: int,
    choice: str,
    lang: str,
    profile: gr.OAuthProfile | None,
):
    """Record the validation, then auto-advance to the next prompt and
    reset the choice radio.

    Returns updates for ``(idx_state, slot_state, current_prompt,
    choice_radio, load_status, save_status)``. On input-validation errors
    the visible inputs are left untouched (``gr.update()``); on successful
    save the next prompt is loaded and the radio cleared so the user has
    an unambiguous cue that the previous validation went through."""
    s = _t(lang)
    keep_state = (
        gr.update(),  # idx_state
        gr.update(),  # slot_state
        gr.update(),  # current_prompt
        gr.update(),  # choice_radio
        gr.update(),  # load_status
    )
    if profile is None:
        return (*keep_state, s["login_required"])
    if idx is None or idx < 0 or slot not in (1, 2, 3):
        return (*keep_state, s["load_first"])
    if choice not in VALIDATION_CHOICES:
        return (*keep_state, s["validation_choice_required"])
    df = load_prompts_df()
    if idx >= len(df):
        return (*keep_state, s["out_of_range"])

    # Defensive: if this user is already a validator on this row (stale
    # form, second browser tab, replayed request, etc.) swallow the
    # duplicate silently — their intent is already fulfilled by the
    # earlier save. Advance to the next prompt as if it had just landed.
    already = any(
        df.at[idx, f"prompt_validation_{i}"]["username"] == profile.username
        for i in (1, 2, 3)
    )
    if already:
        log.info(
            "skipped double-validation: user=%s row=%d", profile.username, idx
        )
    else:
        df.at[idx, f"prompt_validation_{slot}"] = {
            "choice": choice,
            "username": profile.username,
        }
        prompt_id = int(df.at[idx, "id"]) if "id" in df.columns else int(idx)
        push_prompts_df(
            df,
            commit_message=f"{profile.username} validated prompt with ID {prompt_id}",
        )

    # Advance to the next prompt and reset the radio.
    next_idx, next_slot, next_prompt, next_status = fetch_next_validation(
        lang, profile
    )
    return (
        next_idx,                  # idx_state
        next_slot,                 # slot_state
        next_prompt,               # current_prompt
        gr.update(value=None),     # choice_radio reset
        next_status,               # load_status
        s["validation_saved"],     # save_status (same whether we wrote or skipped)
    )


# ---------------------------------------------------------------------------
# Answer voting
# ---------------------------------------------------------------------------


def fetch_next_voting(lang: str, profile: gr.OAuthProfile | None):
    """Return ``(idx, slot, prompt, answer_a, answer_b, status)`` for the next
    fully-validated prompt the user can vote on. ``idx == -1`` means nothing.

    Users *can* vote on their own prompts (unlike validation). Filters to
    the user's own country — voters only judge cultural appropriateness of
    answers grounded in their own culture."""
    s = _t(lang)
    if profile is None:
        return -1, -1, "", "", "", s["login_required"]
    info = participant_info(profile.username)
    user_country = info.get("country") if info else None
    df = load_prompts_df()
    for idx, row in df.iterrows():
        if user_country and row.get("country") != user_country:
            continue
        if not is_fully_validated(row) or not has_answers(row):
            continue
        if any(
            row[f"answer_chosen_{i}"]["username"] == profile.username for i in (1, 2, 3)
        ):
            continue
        for i in (1, 2, 3):
            if not row[f"answer_chosen_{i}"]["username"]:
                display = _merged_prompt_display(
                    lang,
                    row.get("system_prompt", ""),
                    row["prompt"],
                )
                prompt_id = int(row.get("id", idx))
                return (
                    int(idx),
                    i,
                    display,
                    row["answer_a"],
                    row["answer_b"],
                    s["voting_in_progress"].format(
                        id=prompt_id,
                        country=country_display(row.get("country")),
                    ),
                )
    return -1, -1, "", "", "", s["voting_no_more"]


def save_vote(
    idx: int,
    slot: int,
    choice: str,
    lang: str,
    profile: gr.OAuthProfile | None,
):
    """Persist the vote and return the next prompt to vote on."""
    s = _t(lang)
    if profile is None:
        return -1, -1, "", "", "", s["login_required"]
    if idx is None or idx < 0 or slot not in (1, 2, 3) or choice not in VOTE_CHOICES:
        return -1, -1, "", "", "", s["load_first"]
    df = load_prompts_df()
    if idx >= len(df):
        return -1, -1, "", "", "", s["out_of_range"]
    df.at[idx, f"answer_chosen_{slot}"] = {
        "choice": choice,
        "username": profile.username,
    }
    prompt_id = int(df.at[idx, "id"]) if "id" in df.columns else int(idx)
    push_prompts_df(
        df,
        commit_message=f"{profile.username} voted prompt with ID {prompt_id}",
    )
    return fetch_next_voting(lang, profile)


def _vote_handler(choice: str):
    """Build a Gradio handler that records ``choice`` and loads the next prompt.

    A factory is needed so that each click handler keeps its own ``OAuthProfile``
    annotation — Gradio only auto-injects the profile when it sees the
    annotation on the function signature."""

    def handler(idx: int, slot: int, lang: str, profile: gr.OAuthProfile | None):
        return save_vote(idx, slot, choice, lang, profile)

    return handler


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def _build_writing_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    system_box = gr.Textbox(
        label=s["writing_system_label"],
        lines=2,
        placeholder=s["writing_system_placeholder"],
    )
    prompt_box = gr.Textbox(
        label=s["writing_prompt_label"],
        lines=5,
        placeholder=s["writing_prompt_placeholder"],
    )
    save_btn = gr.Button(s["writing_save_button"], variant="primary")
    save_status = gr.Markdown()
    save_btn.click(
        save_prompt,
        inputs=[system_box, prompt_box, language],
        outputs=[save_status, system_box, prompt_box],
    )
    return {
        "system_box": system_box,
        "prompt_box": prompt_box,
        "save_btn": save_btn,
        "save_status": save_status,
    }


def _validation_radio_choices(lang: str) -> list[tuple[str, str]]:
    """`(label, value)` pairs for the validation radio, in display order."""
    s = _t(lang)
    return [(s[f"validation_choice_{c}"], c) for c in VALIDATION_CHOICES]


def _build_validation_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    idx_state = gr.State(-1)
    slot_state = gr.State(-1)
    load_status = gr.Markdown(s["validation_load_status_initial"])
    current_prompt = gr.Textbox(
        label=s["validation_prompt_label"],
        lines=5,
        interactive=False,
    )
    choice_radio = gr.Radio(
        choices=_validation_radio_choices(DEFAULT_LANG),
        label=s["validation_choice_label"],
        value=None,
    )
    with gr.Row():
        load_btn = gr.Button(s["validation_load_button"])
        save_val_btn = gr.Button(s["validation_save_button"], variant="primary")
    save_val_status = gr.Markdown()

    load_btn.click(
        fetch_next_validation,
        inputs=[language],
        outputs=[idx_state, slot_state, current_prompt, load_status],
    )
    save_val_btn.click(
        save_validation,
        inputs=[idx_state, slot_state, choice_radio, language],
        outputs=[
            idx_state,
            slot_state,
            current_prompt,
            choice_radio,
            load_status,
            save_val_status,
        ],
    )
    return {
        "load_status": load_status,
        "current_prompt": current_prompt,
        "choice_radio": choice_radio,
        "load_btn": load_btn,
        "save_btn": save_val_btn,
        "save_status": save_val_status,
    }


def _build_voting_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    idx_state = gr.State(-1)
    slot_state = gr.State(-1)
    status_md = gr.Markdown(s["voting_load_status_initial"])
    current_prompt = gr.Textbox(
        label=s["voting_prompt_label"], lines=4, interactive=False
    )
    with gr.Row():
        ans_a = gr.Textbox(label=s["voting_answer_a_label"], lines=8, interactive=False)
        ans_b = gr.Textbox(label=s["voting_answer_b_label"], lines=8, interactive=False)
    with gr.Row():
        load_btn = gr.Button(s["voting_load_button"])
        a_btn = gr.Button(s["voting_a_button"], variant="primary")
        b_btn = gr.Button(s["voting_b_button"], variant="primary")
        both_btn = gr.Button(s["voting_both_button"])
        none_btn = gr.Button(s["voting_none_button"])

    fetch_outputs = [
        idx_state,
        slot_state,
        current_prompt,
        ans_a,
        ans_b,
        status_md,
    ]
    load_btn.click(fetch_next_voting, inputs=[language], outputs=fetch_outputs)
    for btn, choice in (
        (a_btn, "a"),
        (b_btn, "b"),
        (both_btn, "both"),
        (none_btn, "none"),
    ):
        btn.click(
            _vote_handler(choice),
            inputs=[idx_state, slot_state, language],
            outputs=fetch_outputs,
        )
    return {
        "status_md": status_md,
        "current_prompt": current_prompt,
        "ans_a": ans_a,
        "ans_b": ans_b,
        "load_btn": load_btn,
        "a_btn": a_btn,
        "b_btn": b_btn,
        "both_btn": both_btn,
        "none_btn": none_btn,
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _user_progress_data(stats: dict, lang: str) -> pd.DataFrame:
    """Long-form DataFrame for the user progress bar plot."""
    s = _t(lang)
    return pd.DataFrame(
        {
            "metric": [
                s["plot_metric_sent"],
                s["plot_metric_validated"],
                s["plot_metric_voted"],
            ],
            "count": [stats["sent"], stats["validated"], stats["voted"]],
        }
    )


def _country_plot_data(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """Long-form DataFrame for the stacked country bar plot.

    One row per (country, status) so ``gr.BarPlot(color="status")`` stacks
    fully-validated on top of pending."""
    s = _t(lang)
    counts = country_counts(df)
    if counts.empty:
        return pd.DataFrame(columns=["country", "status", "count"])
    fully = pd.DataFrame(
        {
            "country": counts["country"],
            "status": s["plot_country_fully"],
            "count": counts["fully_validated"].astype(int),
        }
    )
    pending = pd.DataFrame(
        {
            "country": counts["country"],
            "status": s["plot_country_pending"],
            "count": counts["pending"].astype(int),
        }
    )
    return pd.concat([fully, pending], ignore_index=True)


def refresh_leaderboard(lang: str, profile: gr.OAuthProfile | None):
    """Return localized ``(user_plot, ranking, country_plot)`` updates.

    Returns ``gr.update`` payloads (not bare DataFrames) so the BarPlots
    can pick up the language-dependent ``color_map`` and axis titles in the
    same render that brings the data in. Doing the localization here, on
    the lazy tab-open path, also keeps it off the page-load hot path."""
    s = _t(lang)
    df = load_prompts_df()
    username = profile.username if profile else ""
    rdf = ranking_df(df).rename(
        columns={
            "username": s["ranking_col_username"],
            "prompts sent": s["ranking_col_sent"],
            "prompts validated": s["ranking_col_validated"],
            "answers voted": s["ranking_col_voted"],
        }
    )
    return (
        gr.update(
            value=_user_progress_data(user_stats(username, df), lang),
            color_map=_user_progress_color_map(lang),
            x_title=s["plot_xlabel_count"],
        ),
        gr.update(value=rdf),
        gr.update(
            value=_country_plot_data(df, lang),
            color_map=_country_color_map(lang),
            x_title=s["plot_country_xlabel"],
            y_title=s["plot_country_ylabel"],
        ),
    )


def _user_progress_color_map(lang: str) -> dict[str, str]:
    s = _t(lang)
    return {
        s["plot_metric_sent"]: "#3b82f6",
        s["plot_metric_validated"]: "#10b981",
        s["plot_metric_voted"]: "#f59e0b",
    }


def _country_color_map(lang: str) -> dict[str, str]:
    s = _t(lang)
    return {
        s["plot_country_fully"]: "#22c55e",
        s["plot_country_pending"]: "#facc15",
    }


def _build_leaderboard_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    refresh_btn = gr.Button(s["leaderboard_refresh_button"], variant="secondary")
    # Provide an empty DataFrame with the expected columns as initial value.
    # gr.BarPlot referring to columns that don't exist in ``value`` produces a
    # malformed Vega-Lite spec that crashes the client-side renderer and
    # freezes the whole UI — page loads stuck in the static initial render
    # (English Annotation Guidelines, no working login or tabs).
    user_plot = gr.BarPlot(
        value=pd.DataFrame({"metric": [], "count": []}),
        x="count",
        y="metric",
        color="metric",
        color_map=_user_progress_color_map(DEFAULT_LANG),
        x_lim=[0, LEADERBOARD_GOAL],
        x_title=s["plot_xlabel_count"],
        label=s["leaderboard_user_plot_label"].format(goal=LEADERBOARD_GOAL),
        height=180,
    )
    ranking = gr.Dataframe(
        label=s["leaderboard_ranking_label"],
        interactive=False,
        wrap=True,
    )
    country_plot = gr.BarPlot(
        value=pd.DataFrame({"country": [], "count": [], "status": []}),
        x="country",
        y="count",
        color="status",
        color_map=_country_color_map(DEFAULT_LANG),
        x_title=s["plot_country_xlabel"],
        y_title=s["plot_country_ylabel"],
        label=s["leaderboard_country_plot_label"],
        height=320,
    )
    outputs = [user_plot, ranking, country_plot]
    refresh_btn.click(refresh_leaderboard, inputs=[language], outputs=outputs)
    return {
        "refresh_btn": refresh_btn,
        "user_plot": user_plot,
        "ranking": ranking,
        "country_plot": country_plot,
        "outputs": outputs,
    }


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------


def _resolve_language(profile: gr.OAuthProfile | None) -> str:
    """Pick the UI language for ``profile`` from the participants dataset.

    Logged-out visitors and unknown users get ``DEFAULT_LANG``."""
    if profile is None:
        return DEFAULT_LANG
    info = participant_info(profile.username)
    if info and info.get("language") in T:
        return info["language"]
    return DEFAULT_LANG


def init_ui(profile: gr.OAuthProfile | None):
    """Resolve the user's language and emit one update per translatable
    component. Returned tuple layout matches the ``demo.load`` ``outputs=``
    list in :func:`build_demo`.

    The leaderboard's data is *not* computed here — it's loaded lazily when
    the Leaderboard tab is opened (see ``tab_leaderboard.select``). Init only
    updates leaderboard labels."""
    lang = _resolve_language(profile)
    s = _t(lang)
    return (
        # Language state (drives every subsequent handler)
        lang,
        # Top bar
        gr.update(value=s["title"]),
        gr.update(value=s["login_button"]),
        gr.update(value=show_user(lang, profile)),
        # Tabs
        gr.update(label=s["tab_guidelines"]),
        gr.update(label=s["tab_writing"]),
        gr.update(label=s["tab_validation"]),
        gr.update(label=s["tab_voting"]),
        gr.update(label=s["tab_leaderboard"]),
        # Guidelines body
        gr.update(value=_read_guidelines(lang)),
        # Writing
        gr.update(
            label=s["writing_system_label"],
            placeholder=s["writing_system_placeholder"],
        ),
        gr.update(
            label=s["writing_prompt_label"],
            placeholder=s["writing_prompt_placeholder"],
        ),
        gr.update(value=s["writing_save_button"]),
        # Validation (labels + initial status)
        gr.update(value=s["validation_load_status_initial"]),
        gr.update(label=s["validation_prompt_label"]),
        gr.update(
            choices=_validation_radio_choices(lang),
            label=s["validation_choice_label"],
            value=None,
        ),
        gr.update(value=s["validation_load_button"]),
        gr.update(value=s["validation_save_button"]),
        # Voting (labels + initial status)
        gr.update(value=s["voting_load_status_initial"]),
        gr.update(label=s["voting_prompt_label"]),
        gr.update(label=s["voting_answer_a_label"]),
        gr.update(label=s["voting_answer_b_label"]),
        gr.update(value=s["voting_load_button"]),
        gr.update(value=s["voting_a_button"]),
        gr.update(value=s["voting_b_button"]),
        gr.update(value=s["voting_both_button"]),
        gr.update(value=s["voting_none_button"]),
        # Leaderboard: button + component labels only. Data, axis titles and
        # color_map are applied by ``refresh_leaderboard`` when the tab is
        # opened — pushing them here against a value-less BarPlot can crash
        # the client-side Vega renderer and freeze the whole UI.
        gr.update(value=s["leaderboard_refresh_button"]),
        gr.update(
            label=s["leaderboard_user_plot_label"].format(goal=LEADERBOARD_GOAL),
        ),
        gr.update(label=s["leaderboard_ranking_label"]),
        gr.update(label=s["leaderboard_country_plot_label"]),
    )


def build_demo() -> gr.Blocks:
    s = _t(DEFAULT_LANG)
    with gr.Blocks(title="Hackathon 2026") as demo:
        # Hidden state: the resolved language code, written by ``init_ui`` on
        # page load and read by every handler that needs to localize a
        # status message.
        language = gr.State(DEFAULT_LANG)

        title_md = gr.Markdown(s["title"])

        with gr.Row():
            login_btn = gr.LoginButton(value=s["login_button"])
            user_md = gr.Markdown(s["not_logged_in_short"])

        with gr.Tabs():
            tab_guidelines = gr.Tab(s["tab_guidelines"])
            with tab_guidelines:
                # ``sanitize_html=False`` lets the guidelines render the
                # inline ``<img>`` (infographics) and ``<center><a>`` (CTA
                # buttons) tags. The markdown is repo-controlled content,
                # not user input, so disabling sanitization is safe.
                guidelines_md = gr.Markdown(
                    _read_guidelines(DEFAULT_LANG),
                    sanitize_html=False,
                )
            tab_writing = gr.Tab(s["tab_writing"])
            with tab_writing:
                writing = _build_writing_tab(language)
            tab_validation = gr.Tab(s["tab_validation"])
            with tab_validation:
                validation = _build_validation_tab(language)
            tab_voting = gr.Tab(s["tab_voting"])
            with tab_voting:
                voting = _build_voting_tab(language)
            tab_leaderboard = gr.Tab(s["tab_leaderboard"])
            with tab_leaderboard:
                leaderboard = _build_leaderboard_tab(language)

        # Lazy: only hit the prompts dataset + build the leaderboard when the
        # user actually opens the tab, not on every page load.
        tab_leaderboard.select(
            refresh_leaderboard,
            inputs=[language],
            outputs=leaderboard["outputs"],
        )

        demo.load(
            init_ui,
            inputs=None,
            outputs=[
                language,
                title_md,
                login_btn,
                user_md,
                tab_guidelines,
                tab_writing,
                tab_validation,
                tab_voting,
                tab_leaderboard,
                guidelines_md,
                writing["system_box"],
                writing["prompt_box"],
                writing["save_btn"],
                validation["load_status"],
                validation["current_prompt"],
                validation["choice_radio"],
                validation["load_btn"],
                validation["save_btn"],
                voting["status_md"],
                voting["current_prompt"],
                voting["ans_a"],
                voting["ans_b"],
                voting["load_btn"],
                voting["a_btn"],
                voting["b_btn"],
                voting["both_btn"],
                voting["none_btn"],
                leaderboard["refresh_btn"],
                leaderboard["user_plot"],
                leaderboard["ranking"],
                leaderboard["country_plot"],
            ],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()
