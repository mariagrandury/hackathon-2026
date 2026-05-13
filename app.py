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
import random

import gradio as gr
import pandas as pd

log = logging.getLogger("hackathon")

from data import (
    EMPTY_VALIDATION,
    EMPTY_VOTE,
    TEST_PASS_THRESHOLD,
    best_test_score,
    country_counts,
    country_display,
    has_answers,
    is_fully_validated,
    load_prompts_df,
    participant_info,
    push_prompts_df,
    ranking_df,
    record_test_attempt,
    user_stats,
)
from test_data import grade as grade_test
from test_data import load_questions as load_test_questions

VOTE_CHOICES = ("a", "b", "both", "none")
# Validation buckets: three reject buckets and the four AlKhamissi et al.
# (2025) cultural dimensions used as accept buckets. The validation tab shows
# them as two side-by-side radios (Reject | Accept). `data.is_fully_validated`
# treats any of the four accept choices as a positive validation.
REJECT_CHOICES = ("trivial", "stereotype", "unrelated")
ACCEPT_CHOICES = ("knowledge", "preference", "dynamics", "bias_probe")
VALIDATION_CHOICES = REJECT_CHOICES + ACCEPT_CHOICES
LEADERBOARD_GOAL = 100
DEFAULT_LANG = "en"
GUIDELINES_DIR = "guidelines"
IMAGES_DIR = "images"

# Gradio's radio options container (`.wrap`) defaults to `flex-direction: row`,
# so buckets flow side-by-side. Force one bucket per line in each of the two
# validation columns (Reject | Accept).
APP_CSS = (
    ".validation-choices .wrap { flex-direction: column; align-items: flex-start; }"
)

# Fixed-size pool of radio components for the test tab. The tab populates
# the first N at runtime from the current language's question bank and
# hides the rest. Keep this >= the longest per-language test.
MAX_TEST_QUESTIONS = 30

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
        "title": "# Hackathon SomosNLP 2026: Cultural Preferences Challenge",
        "login_button": "Sign in with Hugging Face",
        "not_logged_in": "Not logged in. Click **Sign in with Hugging Face** to start.",
        "not_logged_in_short": "Not logged in.",
        "logged_in_as": "Logged in as **{username}**.",
        "tab_guidelines": "Annotation Guidelines",
        "tab_test": "Entry Test",
        "tab_writing": "Prompt Writing",
        "tab_validation": "Prompt Validation",
        "tab_voting": "Answer Voting",
        "tab_leaderboard": "Leaderboard",
        "guidelines_missing": "Guidelines for this language are not available yet.",
        # Test
        "test_intro": "Answer all questions, then click **Submit test**. You need **{threshold}%** correct to unlock the rest of the app. Unlimited retries — if you fail, reread the guidelines and try again.",
        "test_login_required": "Please log in with Hugging Face to take the test.",
        "test_not_participant": "User `{username}` is not registered as a hackathon participant. Ask the organisers to add you before taking the test.",
        "test_no_questions": "No test is available for this language yet.",
        "test_question_label": "Question {n}: {prompt}",
        "test_submit_button": "Submit test",
        "test_retake_button": "Take the test again",
        "test_status_taken": "You scored **{percent}%** on attempt {attempt}. Pass mark is **{threshold}%**.",
        "test_status_passed": "🎉 You scored **{percent}%** on attempt {attempt}. The Prompt Writing, Validation and Voting tabs are now unlocked.",
        "test_status_failed": "You scored **{percent}%** on attempt {attempt}. You need **{threshold}%** to unlock the rest of the app. Reread the guidelines and take the test again.",
        "test_status_unanswered": "Please answer every question before submitting.",
        "test_status_best": "Your best score so far: **{percent}%**.",
        # Common
        "login_required": "Please log in with Hugging Face first.",
        "load_first": "Load a prompt first.",
        "out_of_range": "Prompt index out of range — try loading a new one.",
        # Writing
        "writing_system_label": "System prompt (optional)",
        "writing_system_placeholder": 'Steering instructions for the model, e.g. "Respond in Spanish without inventing data."',
        # Used both as the placeholder shown when the system_prompt box is
        # empty AND as the value substituted in if the user clicks Save
        # without filling it in. ``{country}`` is interpolated with the
        # display name of the user's country.
        "default_system_prompt": "You are a person from {country}. Respond in Spanish concisely.",
        "writing_prompt_label": "Your prompt",
        "writing_prompt_placeholder": "Write a culturally-grounded prompt…",
        "writing_save_button": "Save",
        "writing_empty": "Prompt cannot be empty.",
        "writing_not_participant": "User `{username}` is not registered as a hackathon participant. Ask the organisers to add you.",
        "writing_saved": "Prompt saved. Thanks!",
        # Validation
        "validation_load_status_initial": "Click **Load next prompt** to start validating.",
        "validation_prompt_label": "Prompt to validate",
        "validation_choice_label": "**How would you classify this prompt, according to the guide?**",
        "validation_group_reject": "Reject",
        "validation_group_accept": "Accept",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Reproduces a stereotype",
        "validation_choice_unrelated": "Not culturally grounded in the country",
        "validation_choice_knowledge": "Cultural knowledge",
        "validation_choice_preference": "Cultural preference / norm",
        "validation_choice_dynamics": "Cultural dynamics / interaction",
        "validation_choice_bias_probe": "Bias probe: neutral prompt that surfaces stereotypes",
        "validation_choice_required": "Select one option (reject or accept) before saving.",
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
        "title": "# Hackathon SomosNLP 2026: Reto Preferencias Culturales",
        "login_button": "Iniciar sesión con Hugging Face",
        "not_logged_in": "Sesión no iniciada. Haz clic en **Iniciar sesión con Hugging Face** para empezar.",
        "not_logged_in_short": "Sesión no iniciada.",
        "logged_in_as": "Sesión iniciada como **{username}**.",
        "tab_guidelines": "Guía de anotación",
        "tab_test": "Test de acceso",
        "tab_writing": "Escribir prompts",
        "tab_validation": "Validar prompts",
        "tab_voting": "Votar respuestas",
        "tab_leaderboard": "Ranking",
        "guidelines_missing": "La guía en este idioma todavía no están disponibles.",
        # Test
        "test_intro": "Responde a todas las preguntas y haz clic en **Enviar test**. Necesitas un **{threshold}%** de aciertos para desbloquear el resto de la aplicación. Puedes reintentarlo cuantas veces quieras — si suspendes, relee la guía y vuelve a intentarlo.",
        "test_login_required": "Por favor, inicia sesión con Hugging Face para hacer el test.",
        "test_not_participant": "El usuario `{username}` no está registrado como participante del hackathon. Pide a los organizadores que te añadan antes de hacer el test.",
        "test_no_questions": "Todavía no hay un test disponible en este idioma.",
        "test_question_label": "Pregunta {n}: {prompt}",
        "test_submit_button": "Enviar test",
        "test_retake_button": "Volver a hacer el test",
        "test_status_taken": "Has obtenido un **{percent}%** en el intento {attempt}. Nota de corte: **{threshold}%**.",
        "test_status_passed": "🎉 Has obtenido un **{percent}%** en el intento {attempt}. Las pestañas de Escribir, Validar y Votar ya están desbloqueadas.",
        "test_status_failed": "Has obtenido un **{percent}%** en el intento {attempt}. Necesitas un **{threshold}%** para desbloquear el resto de la aplicación. Relee la guía y vuelve a intentarlo.",
        "test_status_unanswered": "Por favor, responde a todas las preguntas antes de enviar el test.",
        "test_status_best": "Tu mejor puntuación hasta ahora: **{percent}%**.",
        # Common
        "login_required": "Por favor, inicia sesión con Hugging Face primero.",
        "load_first": "Carga un prompt primero.",
        "out_of_range": "Índice de prompt fuera de rango — intenta cargar uno nuevo.",
        # Writing
        "writing_system_label": "System prompt",
        "writing_system_placeholder": 'Instrucciones para el modelo, p. ej. "Responde en español sin inventar datos."',
        "default_system_prompt": "Eres una persona de {country}. Responde en español de manera concisa.",
        "writing_prompt_label": "Tu prompt",
        "writing_prompt_placeholder": "Escribe un prompt con base cultural…",
        "writing_save_button": "Guardar",
        "writing_empty": "El prompt no puede estar vacío.",
        "writing_not_participant": "El usuario `{username}` no está registrado como participante del hackathon. Pide a los organizadores que te añadan.",
        "writing_saved": "Prompt guardado. ¡Gracias!",
        # Validation
        "validation_load_status_initial": "Haz clic en **Cargar siguiente prompt** para empezar a validar.",
        "validation_prompt_label": "Prompt a validar",
        "validation_choice_label": "**¿Cómo clasificarías este prompt según la guía?**",
        "validation_group_reject": "Rechazar",
        "validation_group_accept": "Aceptar",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Reproduce un estereotipo",
        "validation_choice_unrelated": "Sin anclaje cultural en el país",
        "validation_choice_knowledge": "Conocimiento cultural",
        "validation_choice_preference": "Preferencia o norma cultural",
        "validation_choice_dynamics": "Dinámica cultural / interacción",
        "validation_choice_bias_probe": "Trampa de sesgo: prompt neutral que detecta estereotipos",
        "validation_choice_required": "Selecciona una opción (rechazar o aceptar) antes de guardar.",
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
        "title": "# Hackathon SomosNLP 2026: Preferências Culturais",
        "login_button": "Entrar com o Hugging Face",
        "not_logged_in": "Sessão não iniciada. Clique em **Entrar com o Hugging Face** para começar.",
        "not_logged_in_short": "Sessão não iniciada.",
        "logged_in_as": "Sessão iniciada como **{username}**.",
        "tab_guidelines": "Diretrizes de anotação",
        "tab_test": "Teste de acesso",
        "tab_writing": "Escrever prompts",
        "tab_validation": "Validar prompts",
        "tab_voting": "Votar respostas",
        "tab_leaderboard": "Classificação",
        "guidelines_missing": "As diretrizes neste idioma ainda não estão disponíveis.",
        # Test
        "test_intro": "Responda todas as perguntas e clique em **Enviar teste**. Você precisa de **{threshold}%** de acertos para desbloquear o resto da aplicação. Tentativas ilimitadas — se reprovar, releia as diretrizes e tente novamente.",
        "test_login_required": "Por favor, entre com o Hugging Face para fazer o teste.",
        "test_not_participant": "O usuário `{username}` não está registrado como participante do hackathon. Peça aos organizadores para adicioná-lo antes de fazer o teste.",
        "test_no_questions": "Ainda não há um teste disponível neste idioma.",
        "test_question_label": "Pergunta {n}: {prompt}",
        "test_submit_button": "Enviar teste",
        "test_retake_button": "Refazer o teste",
        "test_status_taken": "Você obteve **{percent}%** na tentativa {attempt}. Nota de corte: **{threshold}%**.",
        "test_status_passed": "🎉 Você obteve **{percent}%** na tentativa {attempt}. As abas Escrever, Validar e Votar foram desbloqueadas.",
        "test_status_failed": "Você obteve **{percent}%** na tentativa {attempt}. Precisa de **{threshold}%** para desbloquear o resto da aplicação. Releia as diretrizes e tente novamente.",
        "test_status_unanswered": "Por favor, responda todas as perguntas antes de enviar o teste.",
        "test_status_best": "Sua melhor pontuação até agora: **{percent}%**.",
        # Common
        "login_required": "Por favor, entre com o Hugging Face primeiro.",
        "load_first": "Carregue um prompt primeiro.",
        "out_of_range": "Índice do prompt fora do intervalo — tente carregar um novo.",
        # Writing
        "writing_system_label": "Mensagem de sistema (opcional)",
        "writing_system_placeholder": 'Instruções para o modelo, p. ex. "Responda em português sem inventar dados."',
        "default_system_prompt": "Você é uma pessoa de {country}. Responda em português de forma concisa.",
        "writing_prompt_label": "Seu prompt",
        "writing_prompt_placeholder": "Escreva um prompt culturalmente fundamentado…",
        "writing_save_button": "Salvar",
        "writing_empty": "O prompt não pode estar vazio.",
        "writing_not_participant": "O usuário `{username}` não está registrado como participante do hackathon. Peça aos organizadores para adicioná-lo.",
        "writing_saved": "Prompt salvo. Obrigado!",
        # Validation
        "validation_load_status_initial": "Clique em **Carregar próximo prompt** para começar a validar.",
        "validation_prompt_label": "Prompt a validar",
        "validation_choice_label": "**Como você classificaria este prompt segundo o guia?**",
        "validation_group_reject": "Rejeitar",
        "validation_group_accept": "Aceitar",
        "validation_choice_trivial": "Trivial / factual",
        "validation_choice_stereotype": "Reproduz / induz um estereótipo",
        "validation_choice_unrelated": "Sem ancoragem cultural no país",
        "validation_choice_knowledge": "Conhecimento cultural",
        "validation_choice_preference": "Preferência ou norma cultural",
        "validation_choice_dynamics": "Dinâmica cultural / interação",
        "validation_choice_bias_probe": "Armadilha de viés: prompt neutro que detecta estereótipos",
        "validation_choice_required": "Selecione uma opção (rejeitar ou aceitar) antes de salvar.",
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


def _default_system_prompt(lang: str, country_code: str | None) -> str:
    """The text used as the placeholder of the writing-tab system_prompt box,
    and substituted into the saved row when the user leaves it blank.

    Returns "" if we can't resolve a country (logged-out / non-participant
    user) — falling back to the generic placeholder string from the
    translations is the caller's job."""
    if not country_code:
        return ""
    template = _t(lang)["default_system_prompt"]
    return template.format(country=country_display(country_code))


def _merged_prompt_display(lang: str, system_prompt: str, prompt: str) -> str:
    """Format ``(system_prompt, prompt)`` for a single read-only textbox.

    Used by the validation and voting tabs (the user asked for those to be
    merged into one cell). Just two paragraphs separated by a blank line —
    no ``[System prompt]`` / ``[Prompt]`` headers (the visual separation is
    enough; the ``lang`` argument is kept for API stability)."""
    del lang  # no longer needed; kept for callers
    sm = (system_prompt or "").strip()
    p = (prompt or "").strip()
    if not sm:
        return p
    return f"{sm}\n\n{p}"


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
    # If the user left the system_prompt blank, fill it in with the
    # country-aware default ("Eres un asistente de IA de {country}…") —
    # matches the gray placeholder shown in the textbox so the saved value
    # is exactly what the user "sees" before typing.
    sys_text = (system_prompt or "").strip()
    if not sys_text:
        sys_text = _default_system_prompt(lang, info.get("country"))
    new_row = {
        "id": next_id,
        "username": profile.username,
        "language": info["language"],
        "country": info["country"],
        "system_prompt": sys_text,
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
    # Randomize iteration order so two users hitting "Load next" at the
    # same time aren't both handed the very first eligible row, and so a
    # single user clicking "Load next" right after a save doesn't keep
    # seeing the same neighbouring prompts.
    indices = list(df.index)
    random.shuffle(indices)
    for idx in indices:
        row = df.loc[idx]
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
    reject_choice: str | None,
    accept_choice: str | None,
    lang: str,
    profile: gr.OAuthProfile | None,
):
    """Record the validation, then auto-advance to the next prompt and
    reset both choice radios.

    The validation tab has two radios (Reject | Accept) kept mutually
    exclusive, so at most one of ``reject_choice``/``accept_choice`` is set;
    the other is ``None``. Returns updates for ``(idx_state, slot_state,
    current_prompt, reject_radio, accept_radio, load_status, save_status)``.
    On input-validation errors the visible inputs are left untouched
    (``gr.update()``); on successful save the next prompt is loaded and both
    radios cleared so the user has an unambiguous cue that the previous
    validation went through."""
    s = _t(lang)
    keep_state = (
        gr.update(),  # idx_state
        gr.update(),  # slot_state
        gr.update(),  # current_prompt
        gr.update(),  # reject_radio
        gr.update(),  # accept_radio
        gr.update(),  # load_status
    )
    choice = accept_choice or reject_choice
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
        log.info("skipped double-validation: user=%s row=%d", profile.username, idx)
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

    # Advance to the next prompt and reset both radios.
    next_idx, next_slot, next_prompt, next_status = fetch_next_validation(lang, profile)
    return (
        next_idx,  # idx_state
        next_slot,  # slot_state
        next_prompt,  # current_prompt
        gr.update(value=None),  # reject_radio reset
        gr.update(value=None),  # accept_radio reset
        next_status,  # load_status
        s["validation_saved"],  # save_status (same whether we wrote or skipped)
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
    # Randomize iteration order — see note on fetch_next_validation.
    indices = list(df.index)
    random.shuffle(indices)
    for idx in indices:
        row = df.loc[idx]
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
# Entry test
# ---------------------------------------------------------------------------


def _test_threshold_pct() -> int:
    return int(round(TEST_PASS_THRESHOLD * 100))


def _test_radio_choices(lang: str) -> list[tuple[str, str]]:
    """All 7 validation buckets as `(label, value)` pairs for the entry test.

    The test asks the user to classify a prompt into the same buckets as
    the validation tab, but it's a single radio (no Reject/Accept split),
    so we list all seven options together."""
    s = _t(lang)
    return [(s[f"validation_choice_{c}"], c) for c in VALIDATION_CHOICES]


def _shuffle_questions(lang: str) -> list[dict]:
    """Return a shuffled copy of the language's question bank."""
    questions = list(load_test_questions(lang))
    random.shuffle(questions)
    return questions


def _test_radio_updates(
    questions: list[dict],
    lang: str,
    reset_values: bool,
) -> list:
    """Build ``MAX_TEST_QUESTIONS`` update dicts: visible+populated for the
    questions in ``questions``, hidden for the rest. ``reset_values=True``
    clears any previously selected answer (used on (re-)load); ``False``
    leaves the chosen value alone (used after submit so the user can see
    what they picked while reading the result)."""
    s = _t(lang)
    choices = _test_radio_choices(lang)
    updates = []
    for i in range(MAX_TEST_QUESTIONS):
        if i < len(questions):
            q = questions[i]
            update = dict(
                visible=True,
                choices=choices,
                label=s["test_question_label"].format(n=i + 1, prompt=q["prompt"]),
            )
            if reset_values:
                update["value"] = None
            updates.append(gr.update(**update))
        else:
            updates.append(gr.update(visible=False))
    return updates


def load_test(lang: str, profile: gr.OAuthProfile | None):
    """Shuffle questions and return updates for the test UI.

    Outputs (in order): ``questions_state``, ``intro_md``, ``status_md``,
    ``submit_btn``, ``retake_btn``, then ``MAX_TEST_QUESTIONS`` radios.
    """
    s = _t(lang)
    questions = _shuffle_questions(lang)
    intro = s["test_intro"].format(threshold=_test_threshold_pct())
    if not questions:
        status = s["test_no_questions"]
        return (
            [],
            gr.update(value=intro),
            gr.update(value=status),
            gr.update(visible=False),
            gr.update(visible=False),
            *(gr.update(visible=False) for _ in range(MAX_TEST_QUESTIONS)),
        )
    status_lines = []
    if profile is not None:
        best = best_test_score(profile.username)
        if best > 0:
            status_lines.append(
                s["test_status_best"].format(percent=int(round(best * 100)))
            )
    status = "\n\n".join(status_lines)
    return (
        questions,
        gr.update(value=intro),
        gr.update(value=status),
        gr.update(visible=True),
        gr.update(visible=False),
        *_test_radio_updates(questions, lang, reset_values=True),
    )


def submit_test(
    questions: list[dict],
    lang: str,
    profile: gr.OAuthProfile | None,
    *answers,
):
    """Grade the submitted answers, persist the score, and return UI updates.

    Outputs (in order): ``status_md``, ``submit_btn``, ``retake_btn``,
    ``tab_writing``, ``tab_validation``, ``tab_voting``.
    """
    s = _t(lang)
    threshold_pct = _test_threshold_pct()
    noop = (gr.update(), gr.update(), gr.update())  # tabs unchanged
    if profile is None:
        return (
            gr.update(value=s["test_login_required"]),
            gr.update(visible=True),
            gr.update(visible=False),
            *noop,
        )
    if not questions:
        return (
            gr.update(value=s["test_no_questions"]),
            gr.update(visible=False),
            gr.update(visible=False),
            *noop,
        )
    # Walk in question order — answers come positionally from the radio
    # pool, so the i-th answer belongs to the i-th question.
    paired = [(q["id"], answers[i]) for i, q in enumerate(questions)]
    if any(value is None for _, value in paired):
        return (
            gr.update(value=s["test_status_unanswered"]),
            gr.update(visible=True),
            gr.update(visible=False),
            *noop,
        )
    score, _correct, _total = grade_test(paired, lang)
    try:
        attempt = record_test_attempt(profile.username, score)
    except LookupError:
        return (
            gr.update(value=s["test_not_participant"].format(username=profile.username)),
            gr.update(visible=True),
            gr.update(visible=False),
            *noop,
        )
    percent = int(round(score * 100))
    passed = score >= TEST_PASS_THRESHOLD
    if passed:
        status = s["test_status_passed"].format(percent=percent, attempt=attempt)
    else:
        status = s["test_status_failed"].format(
            percent=percent, attempt=attempt, threshold=threshold_pct
        )
    # Reveal the gated tabs in response — the user doesn't need to reload
    # the page to gain access after passing. We don't *hide* them on failure,
    # because the user might have already passed on a previous attempt with
    # a higher score; visibility for non-passers is set on page load.
    tab_update = gr.update(visible=True) if passed else gr.update()
    return (
        gr.update(value=status),
        gr.update(visible=False),
        gr.update(visible=True),
        tab_update,
        tab_update,
        tab_update,
    )


def _build_test_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    intro_md = gr.Markdown(
        s["test_intro"].format(threshold=_test_threshold_pct())
    )
    questions_state = gr.State([])
    radios: list[gr.Radio] = []
    for _ in range(MAX_TEST_QUESTIONS):
        radios.append(
            gr.Radio(
                choices=_test_radio_choices(DEFAULT_LANG),
                value=None,
                visible=False,
                interactive=True,
            )
        )
    submit_btn = gr.Button(s["test_submit_button"], variant="primary", visible=False)
    status_md = gr.Markdown()
    retake_btn = gr.Button(s["test_retake_button"], visible=False)
    return {
        "intro_md": intro_md,
        "questions_state": questions_state,
        "radios": radios,
        "submit_btn": submit_btn,
        "status_md": status_md,
        "retake_btn": retake_btn,
    }


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


def _validation_reject_choices(lang: str) -> list[tuple[str, str]]:
    """`(label, value)` pairs for the Reject radio, in display order."""
    s = _t(lang)
    return [(s[f"validation_choice_{c}"], c) for c in REJECT_CHOICES]


def _validation_accept_choices(lang: str) -> list[tuple[str, str]]:
    """`(label, value)` pairs for the Accept radio, in display order."""
    s = _t(lang)
    return [(s[f"validation_choice_{c}"], c) for c in ACCEPT_CHOICES]


def _clear_other_radio(this_value: str | None):
    """Mutual-exclusivity helper for the two validation radios.

    Wired to each radio's ``.change``: when this radio gains a value, clear
    the other; when it's cleared, leave the other alone (``gr.update()`` is a
    no-op, so it doesn't re-trigger the other radio's ``.change`` — that's
    what stops an infinite clear-each-other cascade)."""
    return gr.update(value=None) if this_value else gr.update()


def _build_validation_tab(language: gr.State) -> dict:
    s = _t(DEFAULT_LANG)
    idx_state = gr.State(-1)
    slot_state = gr.State(-1)
    load_status = gr.Markdown(s["validation_load_status_initial"])
    current_prompt = gr.Textbox(
        label=s["validation_prompt_label"],
        lines=8,
        interactive=False,
    )
    # The seven buckets are split across two side-by-side radios so the
    # annotator sees Reject vs Accept as distinct columns. They behave as a
    # single choice: ``_clear_other_radio`` keeps at most one selected, and
    # ``save_validation`` reads whichever holds a value.
    choice_label = gr.Markdown(s["validation_choice_label"])
    with gr.Row():
        with gr.Column():
            reject_radio = gr.Radio(
                choices=_validation_reject_choices(DEFAULT_LANG),
                label=s["validation_group_reject"],
                value=None,
                elem_classes=["validation-choices"],
            )
        with gr.Column():
            accept_radio = gr.Radio(
                choices=_validation_accept_choices(DEFAULT_LANG),
                label=s["validation_group_accept"],
                value=None,
                elem_classes=["validation-choices"],
            )
    with gr.Row():
        load_btn = gr.Button(s["validation_load_button"])
        save_val_btn = gr.Button(s["validation_save_button"], variant="primary")
    save_val_status = gr.Markdown()

    reject_radio.change(
        _clear_other_radio, inputs=[reject_radio], outputs=[accept_radio]
    )
    accept_radio.change(
        _clear_other_radio, inputs=[accept_radio], outputs=[reject_radio]
    )

    load_btn.click(
        fetch_next_validation,
        inputs=[language],
        outputs=[idx_state, slot_state, current_prompt, load_status],
    )
    save_val_btn.click(
        save_validation,
        inputs=[idx_state, slot_state, reject_radio, accept_radio, language],
        outputs=[
            idx_state,
            slot_state,
            current_prompt,
            reject_radio,
            accept_radio,
            load_status,
            save_val_status,
        ],
    )
    return {
        "load_status": load_status,
        "current_prompt": current_prompt,
        "choice_label": choice_label,
        "reject_radio": reject_radio,
        "accept_radio": accept_radio,
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
        label=s["voting_prompt_label"], lines=8, interactive=False
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
    # Country-aware placeholder for the system_prompt textbox — uses the
    # actual default that will be substituted in if the user clicks Save
    # without filling it in.
    info = participant_info(profile.username) if profile else None
    default_sys = _default_system_prompt(lang, info.get("country") if info else None)
    sys_placeholder = default_sys or s["writing_system_placeholder"]
    # Gating: Writing / Validation / Voting only become available once the
    # user has scored above ``TEST_PASS_THRESHOLD`` on the entry test.
    passed = (
        best_test_score(profile.username) >= TEST_PASS_THRESHOLD
        if profile is not None
        else False
    )
    test_state = load_test(lang, profile)
    return (
        # Language state (drives every subsequent handler)
        lang,
        # Top bar
        gr.update(value=s["title"]),
        gr.update(value=s["login_button"]),
        gr.update(value=show_user(lang, profile)),
        # Tabs (gated tabs get visible=passed)
        gr.update(label=s["tab_guidelines"]),
        gr.update(label=s["tab_test"]),
        gr.update(label=s["tab_writing"], visible=passed),
        gr.update(label=s["tab_validation"], visible=passed),
        gr.update(label=s["tab_voting"], visible=passed),
        gr.update(label=s["tab_leaderboard"]),
        # Guidelines body
        gr.update(value=_read_guidelines(lang)),
        # Test: questions_state, intro, status, submit_btn, retake_btn, *radios
        *test_state,
        # Writing
        gr.update(
            label=s["writing_system_label"],
            placeholder=sys_placeholder,
        ),
        gr.update(
            label=s["writing_prompt_label"],
            placeholder=s["writing_prompt_placeholder"],
        ),
        gr.update(value=s["writing_save_button"]),
        # Validation (labels + initial status)
        gr.update(value=s["validation_load_status_initial"]),
        gr.update(label=s["validation_prompt_label"]),
        gr.update(value=s["validation_choice_label"]),
        gr.update(
            choices=_validation_reject_choices(lang),
            label=s["validation_group_reject"],
            value=None,
        ),
        gr.update(
            choices=_validation_accept_choices(lang),
            label=s["validation_group_accept"],
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
    with gr.Blocks(title="Hackathon 2026", css=APP_CSS) as demo:
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
            tab_test = gr.Tab(s["tab_test"])
            with tab_test:
                test_tab = _build_test_tab(language)
            # Gated tabs — only visible once the user has passed the entry test.
            tab_writing = gr.Tab(s["tab_writing"], visible=False)
            with tab_writing:
                writing = _build_writing_tab(language)
            tab_validation = gr.Tab(s["tab_validation"], visible=False)
            with tab_validation:
                validation = _build_validation_tab(language)
            tab_voting = gr.Tab(s["tab_voting"], visible=False)
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

        # Test handlers — Submit grades and toggles gated-tab visibility;
        # Retake re-shuffles and resets the radios.
        test_tab["submit_btn"].click(
            submit_test,
            inputs=[test_tab["questions_state"], language, *test_tab["radios"]],
            outputs=[
                test_tab["status_md"],
                test_tab["submit_btn"],
                test_tab["retake_btn"],
                tab_writing,
                tab_validation,
                tab_voting,
            ],
        )
        test_tab["retake_btn"].click(
            load_test,
            inputs=[language],
            outputs=[
                test_tab["questions_state"],
                test_tab["intro_md"],
                test_tab["status_md"],
                test_tab["submit_btn"],
                test_tab["retake_btn"],
                *test_tab["radios"],
            ],
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
                tab_test,
                tab_writing,
                tab_validation,
                tab_voting,
                tab_leaderboard,
                guidelines_md,
                test_tab["questions_state"],
                test_tab["intro_md"],
                test_tab["status_md"],
                test_tab["submit_btn"],
                test_tab["retake_btn"],
                *test_tab["radios"],
                writing["system_box"],
                writing["prompt_box"],
                writing["save_btn"],
                validation["load_status"],
                validation["current_prompt"],
                validation["choice_label"],
                validation["reject_radio"],
                validation["accept_radio"],
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
