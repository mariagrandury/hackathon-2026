"""End-to-end integration test for the app, with the HF I/O layer stubbed.

Runs every user-facing handler against an in-memory fake of the prompts
dataset (so "fake sending prompts, saving, fetching" never touches HF),
reports per-call wall-clock latency, and asserts the invariants you'd
expect (the saved row actually shows up in the next fetch, the picker
skips your own prompts, voting blocks until 3 ``relevant`` validations,
etc.). Run with::

    python test_integration.py
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import pandas as pd
from dotenv import load_dotenv

# Load .env so the OAuth mock can call ``whoami()`` with a real token —
# Gradio runs that at Blocks.__exit__ time when ``gr.LoginButton`` is in
# the tree, even though we never use OAuth in this test. The HF I/O calls
# themselves are stubbed below.
load_dotenv()

# Quiet the per-handler INFO logs from the app.
logging.getLogger("hackathon").setLevel(logging.WARNING)

import data  # noqa: E402

# ---------------------------------------------------------------------------
# Step 1 — in-memory fakes for the HF I/O layer.
# ---------------------------------------------------------------------------

PARTICIPANTS_DF = pd.DataFrame(
    [
        {
            "username": "mariagrandury",
            "language": "es",
            "country": "es",
            "gmail": "maria@x",
        },
        {"username": "alice-cl", "language": "es", "country": "cl", "gmail": "alice@x"},
        {"username": "bruno-br", "language": "pt", "country": "br", "gmail": "bruno@x"},
        {"username": "carla-co", "language": "es", "country": "co", "gmail": "carla@x"},
        {"username": "diogo-pt", "language": "pt", "country": "pt", "gmail": "diogo@x"},
    ]
)

# Mutable container — push_prompts_df reassigns into this dict so a later
# load_prompts_df sees the updated state. Has to be a container (not a
# bare reassignment) so the closures below capture-by-reference.
STATE: dict = {"prompts": None}


def _val(user: str, choice: str = "relevant") -> dict:
    return {"choice": choice, "username": user}


def _empty_val() -> dict:
    return {"choice": "", "username": ""}


def _empty_vote() -> dict:
    return {"choice": "", "username": ""}


def _seed_prompts() -> pd.DataFrame:
    """A hand-crafted 6-row set covering every code path we want to test."""
    rows = [
        # 0 — pristine v0-style prompt with system_prompt and both answers.
        {
            "id": 1,
            "username": "v0",
            "language": "es",
            "country": "es",
            "system_prompt": "Una conversación entre un usuario y un modelo de lenguaje. Responde en español sin inventar datos.",
            "prompt": "¿Qué se cena en Nochevieja en España?",
            "prompt_validation_1": _empty_val(),
            "prompt_validation_2": _empty_val(),
            "prompt_validation_3": _empty_val(),
            "answer_a": "Las 12 uvas a medianoche, langostinos y turrón.",
            "model_a": "gpt-4o-mini",
            "answer_b": "Bacalao a la vizcaína y cordero asado.",
            "model_b": "claude-3-5-sonnet",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
        # 1 — partially validated (1 of 3).
        {
            "username": "v0",
            "language": "es",
            "country": "cl",
            "system_prompt": "",
            "prompt": "¿Qué es una 'completada' en Chile?",
            "prompt_validation_1": _val("bruno-br"),
            "prompt_validation_2": _empty_val(),
            "prompt_validation_3": _empty_val(),
            "answer_a": "Una reunión informal para comer completos (hot dogs).",
            "model_a": "mistral-large-2411",
            "answer_b": "Es un plato chileno que combina pan y palta.",
            "model_b": "gemini-2.0-flash-exp",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
        # 2 — already fully validated, answers present, no votes yet.
        {
            "username": "v0",
            "language": "pt",
            "country": "br",
            "system_prompt": "Responda em português brasileiro.",
            "prompt": "Qual é o prato típico do São João?",
            "prompt_validation_1": _val("alice-cl"),
            "prompt_validation_2": _val("carla-co"),
            "prompt_validation_3": _val("diogo-pt"),
            "answer_a": "Canjica, pamonha e milho cozido.",
            "model_a": "llama-3.2-90b-vision-instruct-maas",
            "answer_b": "Quentão, paçoca e bolo de milho.",
            "model_b": "gpt-4o-mini",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
        # 3 — tie/both_bad style: prompt + system msg but NO answers.
        {
            "username": "v0",
            "language": "es",
            "country": "co",
            "system_prompt": "Responde en español neutral.",
            "prompt": "¿Cuál es el mejor postre de Colombia?",
            "prompt_validation_1": _empty_val(),
            "prompt_validation_2": _empty_val(),
            "prompt_validation_3": _empty_val(),
            "answer_a": "",
            "model_a": "",
            "answer_b": "",
            "model_b": "",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
        # 4 — authored by the test user themselves (should be skipped by
        # the validation picker, but voteable when fully validated).
        {
            "username": "mariagrandury",
            "language": "es",
            "country": "es",
            "system_prompt": "",
            "prompt": "¿Qué se come en una boda gallega?",
            "prompt_validation_1": _val("alice-cl"),
            "prompt_validation_2": _val("bruno-br"),
            "prompt_validation_3": _val("carla-co"),
            "answer_a": "Empanada, pulpo y queso de tetilla.",
            "model_a": "gpt-4o-mini",
            "answer_b": "Cordero, lacón y filloas.",
            "model_b": "claude-3-5-sonnet",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
        # 5 — fully validated but one validator is the test user (so they
        # can't pick it for validation; can vote on it).
        {
            "username": "v0",
            "language": "es",
            "country": "ec",
            "system_prompt": "",
            "prompt": "¿Qué es el encebollado ecuatoriano?",
            "prompt_validation_1": _val("alice-cl"),
            "prompt_validation_2": _val("mariagrandury"),
            "prompt_validation_3": _val("bruno-br"),
            "answer_a": "Una sopa de pescado con yuca y cebolla.",
            "model_a": "mistral-large-2411",
            "answer_b": "Plato típico del litoral, base de albacora y yuca.",
            "model_b": "gemini-2.0-flash-exp",
            "answer_chosen_1": _empty_vote(),
            "answer_chosen_2": _empty_vote(),
            "answer_chosen_3": _empty_vote(),
        },
    ]
    return pd.DataFrame(rows)


STATE["prompts"] = _seed_prompts()


def fake_load_prompts() -> pd.DataFrame:
    return STATE["prompts"].copy()


def fake_load_participants() -> pd.DataFrame:
    return PARTICIPANTS_DF.copy()


push_calls: list[tuple[float, int]] = []


def fake_push(df: pd.DataFrame) -> None:
    STATE["prompts"] = df.copy()
    push_calls.append((time.monotonic(), len(df)))


# Patch BEFORE importing app — app's ``from data import …`` will then pick
# up the stubs instead of the real HF-hitting functions.
data.load_prompts_df = fake_load_prompts
data.load_participants_df = fake_load_participants
data.push_prompts_df = fake_push

import app  # noqa: E402

# Verify the patches stuck.
assert (
    app.load_prompts_df is fake_load_prompts
), "stub didn't take on app.load_prompts_df"
assert app.push_prompts_df is fake_push, "stub didn't take on app.push_prompts_df"


# ---------------------------------------------------------------------------
# Step 2 — fake profile + helpers.
# ---------------------------------------------------------------------------


def profile(username: str) -> SimpleNamespace:
    """Stand-in for ``gr.OAuthProfile`` (only ``.username`` is read)."""
    return SimpleNamespace(username=username)


class Timer:
    def __init__(self):
        self.runs: list[tuple[str, float]] = []

    def time(self, label: str, fn, *args, **kw):
        t0 = time.monotonic()
        result = fn(*args, **kw)
        dt = time.monotonic() - t0
        self.runs.append((label, dt))
        return result, dt

    def report(self) -> None:
        print()
        print(f"  {'Operation':50s} {'Time':>8s}")
        print(f"  {'-'*50} {'-'*8}")
        for label, dt in self.runs:
            print(f"  {label:50s} {dt*1000:6.1f}ms")
        total = sum(dt for _, dt in self.runs)
        print(f"  {'-'*50} {'-'*8}")
        print(f"  {'TOTAL':50s} {total*1000:6.1f}ms")


T = Timer()


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check(name: str, ok: bool, detail: str = "") -> None:
    marker = "✓" if ok else "✗ FAIL"
    print(f"  {marker} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        raise SystemExit(f"Assertion failed: {name}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_demo_builds():
    section("1. demo.build_demo() — Gradio block tree")
    demo, dt = T.time("build_demo (cold)", app.build_demo)
    check("demo builds", demo is not None)
    check(
        "≥ 40 blocks (tabs + components)",
        len(demo.blocks) >= 40,
        f"{len(demo.blocks)} blocks",
    )


def test_init_ui_logged_out():
    section("2. init_ui — logged-out visitor")
    out, dt = T.time("init_ui (profile=None)", app.init_ui, None)
    # init_ui returns a tuple of updates. First element is the language.
    lang = out[0]
    check("language defaults to 'en' when logged out", lang == "en")
    check(f"returns {len(out)} updates", len(out) > 20)


def test_init_ui_known_user():
    section("3. init_ui — known participant (mariagrandury, lang=es)")
    out, _ = T.time("init_ui (mariagrandury)", app.init_ui, profile("mariagrandury"))
    check("resolves language to 'es'", out[0] == "es")
    out2, _ = T.time("init_ui (bruno-br, lang=pt)", app.init_ui, profile("bruno-br"))
    check("resolves language to 'pt' for bruno-br", out2[0] == "pt")


def test_merged_display():
    section("4. _merged_prompt_display helper")
    plain = app._merged_prompt_display("en", "", "the prompt")
    check("empty system_prompt → prompt only", plain == "the prompt")

    merged = app._merged_prompt_display("en", "be brief", "what time is it?")
    check(
        "non-empty system_prompt → labelled merge",
        "[System prompt]" in merged
        and "[Prompt]" in merged
        and "be brief" in merged
        and "what time is it?" in merged,
        merged.replace("\n", "⏎"),
    )


def test_validation_picker_skips():
    section("5. fetch_next_validation — picker behaviour")
    # mariagrandury is the author of row 4 and a validator on row 5,
    # so the picker should NOT return either.
    (idx, slot, prompt_display, status), _ = T.time(
        "fetch_next_validation (mariagrandury)",
        app.fetch_next_validation,
        "es",
        profile("mariagrandury"),
    )
    check(
        "picker returns a valid (idx, slot)",
        idx >= 0 and slot in (1, 2, 3),
        f"got idx={idx}, slot={slot}",
    )
    check("picker skipped row 4 (own prompt)", idx != 4)
    check("picker skipped row 5 (already validated by maria)", idx != 5)
    sm = STATE["prompts"].at[idx, "system_prompt"]
    p = STATE["prompts"].at[idx, "prompt"]
    if sm:
        check(
            "merged display includes the system_prompt text", sm[:40] in prompt_display
        )
        check("merged display includes the prompt text", p[:40] in prompt_display)
        check(
            "system_prompt appears before prompt in display",
            prompt_display.find(sm[:30]) < prompt_display.find(p[:30]),
        )


def test_validation_save_advances_state():
    section("6. save_validation — write + verify state change")
    user = profile("alice-cl")
    (idx0, slot0, *_), _ = T.time(
        "fetch_next_validation (alice-cl)", app.fetch_next_validation, "es", user
    )

    pre = STATE["prompts"].at[idx0, f"prompt_validation_{slot0}"]
    check("target slot empty before save", pre["username"] == "")

    out, _ = T.time(
        "save_validation (relevant)",
        app.save_validation,
        idx0,
        slot0,
        "relevant",
        "es",
        user,
    )
    # save_validation now returns 6 outputs: (idx, slot, prompt, radio_update,
    # load_status, save_status) — UI auto-advances and clears the choice.
    check("returns 6-tuple (auto-advance shape)",
          isinstance(out, tuple) and len(out) == 6)
    radio_upd = out[3]
    check("choice radio is reset (value=None)",
          isinstance(radio_upd, dict) and radio_upd.get("value") is None)
    next_idx_returned = out[0]
    check("auto-advances to a different row in same response",
          next_idx_returned != idx0,
          f"advanced from row {idx0} to row {next_idx_returned}")

    post = STATE["prompts"].at[idx0, f"prompt_validation_{slot0}"]
    check(
        "slot now records alice-cl with choice='relevant'",
        post["username"] == "alice-cl" and post["choice"] == "relevant",
        f"{post}",
    )

    # Picker no longer offers the same row to alice-cl.
    (idx1, slot1, *_), _ = T.time(
        "fetch_next_validation (alice-cl, after save)",
        app.fetch_next_validation,
        "es",
        user,
    )
    check("picker skips the row alice-cl just validated", idx1 != idx0)


def test_save_validation_self_skip():
    section("7. save_validation — own-prompt skipping in picker")
    user = profile("mariagrandury")
    (idx, *_), _ = T.time(
        "fetch_next_validation (mariagrandury) again",
        app.fetch_next_validation,
        "es",
        user,
    )
    # Should never be row 4 (which is mariagrandury's own).
    check("never returns own prompt (row 4)", idx != 4)


def test_voting_flow():
    section("8. fetch_next_voting + save_vote — full flow")
    # mariagrandury can vote on row 2 (fully validated, has answers, she's
    # not a validator there) and row 5 (fully validated, she is a validator
    # there but voting allows that).
    user = profile("mariagrandury")
    (idx, slot, prompt_display, ans_a, ans_b, status), _ = T.time(
        "fetch_next_voting (mariagrandury)", app.fetch_next_voting, "es", user
    )
    check(
        "returns a fully-validated prompt with answers", idx in (2, 5), f"got idx={idx}"
    )
    check("answer_a is non-empty", bool(ans_a))
    check("answer_b is non-empty", bool(ans_b))
    check(
        "models are NOT shown in the voting view",
        "gpt-4o" not in prompt_display
        and "gpt-4o" not in (ans_a + ans_b)[:0],  # only checking display
        "models hidden in display",
    )

    # Cast the vote.
    out, _ = T.time("save_vote (choice='a')", app.save_vote, idx, slot, "a", "es", user)
    # save_vote chains into fetch_next_voting, so returns a 6-tuple.
    check(
        "save_vote returns next voting payload",
        isinstance(out, tuple) and len(out) == 6,
    )

    after = STATE["prompts"].at[idx, f"answer_chosen_{slot}"]
    check(
        "vote slot now records (mariagrandury, 'a')",
        after["username"] == "mariagrandury" and after["choice"] == "a",
        f"{after}",
    )


def test_voting_blocks_unvalidated():
    section("9. voting picker — blocks unvalidated rows")
    # Row 0 is unvalidated. Even though it has answers, voting should not offer it.
    # bruno-br has not yet validated row 1; only row 2 is fully validated +
    # already voted on above. Row 5 should be offered.
    user = profile("bruno-br")
    (idx, *_), _ = T.time(
        "fetch_next_voting (bruno-br)", app.fetch_next_voting, "es", user
    )
    check(
        "returns a fully-validated row with answers", idx in (2, 4, 5), f"got idx={idx}"
    )
    check("never returns row 0 (unvalidated)", idx != 0)
    check("never returns row 1 (partially validated)", idx != 1)
    check("never returns row 3 (no answers)", idx != 3)


def test_save_prompt():
    section("10. save_prompt — Writing tab, with system_prompt")
    user = profile("mariagrandury")
    n_before = len(STATE["prompts"])
    sm = "Responde como un experto local."
    p = "¿Qué se come en la Feria de Abril en Sevilla?"
    out, _ = T.time(
        "save_prompt (with system_prompt)", app.save_prompt, sm, p, "es", user
    )
    # save_prompt now returns (status, system_box_update, prompt_box_update).
    check("returns a 3-tuple (status + 2 textbox updates)",
          isinstance(out, tuple) and len(out) == 3)
    msg = out[0]
    check(
        "returns a 'saved' status",
        "saved" in msg.lower() or "guardado" in msg.lower() or "salvo" in msg.lower(),
        msg,
    )
    # On success the textboxes are cleared via gr.update(value="").
    sys_upd, prompt_upd = out[1], out[2]
    check("system_box update clears the value (value='')",
          getattr(sys_upd, "get", lambda *_: None)("value") == ""
          or (isinstance(sys_upd, dict) and sys_upd.get("value") == ""))
    check("prompt_box update clears the value (value='')",
          (isinstance(prompt_upd, dict) and prompt_upd.get("value") == ""))
    check(
        "row count incremented",
        len(STATE["prompts"]) == n_before + 1,
        f"{n_before} → {len(STATE['prompts'])}",
    )
    new_row = STATE["prompts"].iloc[-1]
    check("new row has the expected prompt", new_row["prompt"] == p)
    check("new row has the system_prompt", new_row["system_prompt"] == sm)
    check(
        "new row is authored by mariagrandury", new_row["username"] == "mariagrandury"
    )
    check(
        "new row inherits language='es' from participant", new_row["language"] == "es"
    )
    check(
        "new row has empty validation slots",
        all(new_row[f"prompt_validation_{i}"]["username"] == "" for i in (1, 2, 3)),
    )


def test_save_prompt_input_validation():
    section("11. save_prompt — input validation")
    user = profile("mariagrandury")
    out_empty, _ = T.time(
        "save_prompt (empty prompt)", app.save_prompt, "", "  ", "es", user
    )
    msg_empty = out_empty[0]
    check(
        "empty prompt → reject",
        "empty" in msg_empty.lower() or "vac" in msg_empty.lower(),
    )
    # On rejection the textboxes are NOT cleared — user can fix and resubmit.
    check(
        "on rejection, textbox updates leave value untouched",
        not isinstance(out_empty[1], dict)
        or "value" not in out_empty[1]
        or out_empty[1].get("value") != "",
    )

    out_anon, _ = T.time(
        "save_prompt (logged out)", app.save_prompt, "", "hola", "es", None
    )
    msg_anon = out_anon[0]
    check(
        "no profile → login_required",
        "log in" in msg_anon.lower() or "inicia" in msg_anon.lower(),
    )


def test_save_prompt_round_trip_in_validation_picker():
    section("12. round trip — saved prompt is offered to other users for validation")
    # alice-cl just saved on row 0, but we want to see if the prompt
    # mariagrandury saved in test 10 shows up for, say, bruno-br.
    user = profile("bruno-br")
    seen_idx = None
    for _ in range(50):  # picker is greedy; iterate a few times in case
        (idx, slot, prompt_display, *_), _dt = T.time(
            "fetch_next_validation (bruno-br) — scanning",
            app.fetch_next_validation,
            "pt",
            user,
        )
        if "Feria de Abril" in prompt_display:
            seen_idx = idx
            break
        if idx == -1:
            break
        # advance by recording a fake validation so the picker moves past it
        df = STATE["prompts"].copy()
        df.at[idx, f"prompt_validation_{slot}"] = {
            "choice": "trivial",
            "username": "bruno-br",
        }
        STATE["prompts"] = df
    check(
        "the just-saved prompt was eventually offered for validation",
        seen_idx is not None,
        f"seen_idx={seen_idx}",
    )


def test_leaderboard():
    section("13. refresh_leaderboard — aggregations")
    user = profile("mariagrandury")
    out, dt = T.time(
        "refresh_leaderboard (mariagrandury)", app.refresh_leaderboard, "es", user
    )
    # Returns gr.update objects or DataFrames; verify shape.
    check(
        "returns a 3-tuple (user_plot, ranking, country_plot)",
        isinstance(out, tuple) and len(out) == 3,
    )


def test_full_validation_unlocks_voting():
    section("14. invariant — 3 'relevant' validations unlock voting")
    # Take row 1 (partially validated by bruno-br) and complete it.
    idx = 1
    for user, slot in (("alice-cl", 2), ("carla-co", 3)):
        msg, _ = T.time(
            f"save_validation ({user}) on row 1",
            app.save_validation,
            idx,
            slot,
            "relevant",
            "es",
            profile(user),
        )
    # Now row 1 should be fully validated. Check that it's now offered.
    final = STATE["prompts"].at[idx, "prompt_validation_3"]
    check(
        "row 1 now fully validated",
        final["username"] == "carla-co" and final["choice"] == "relevant",
    )

    # Confirm it's now in the voting pool.
    user = profile("alice-cl")  # alice is allowed to vote on her own
    seen = False
    for _ in range(20):
        (vidx, *_), _ = T.time(
            "fetch_next_voting (alice-cl) — scanning", app.fetch_next_voting, "es", user
        )
        if vidx == 1:
            seen = True
            break
        if vidx == -1:
            break
        # mark as voted on so picker advances
        df = STATE["prompts"].copy()
        for s in (1, 2, 3):
            if not df.at[vidx, f"answer_chosen_{s}"]["username"]:
                df.at[vidx, f"answer_chosen_{s}"] = {
                    "choice": "a",
                    "username": "alice-cl",
                }
                break
        STATE["prompts"] = df
    check("voting picker now offers freshly-validated row 1", seen)


def main() -> None:
    t_start = time.monotonic()
    test_demo_builds()
    test_init_ui_logged_out()
    test_init_ui_known_user()
    test_merged_display()
    test_validation_picker_skips()
    test_validation_save_advances_state()
    test_save_validation_self_skip()
    test_voting_flow()
    test_voting_blocks_unvalidated()
    test_save_prompt()
    test_save_prompt_input_validation()
    test_save_prompt_round_trip_in_validation_picker()
    test_leaderboard()
    test_full_validation_unlocks_voting()
    t_total = time.monotonic() - t_start

    print()
    print("=" * 72)
    print(f"  Total push_prompts_df calls: {len(push_calls)}")
    print(
        f"  Final dataset size: {len(STATE['prompts'])} rows "
        f"(started at {len(_seed_prompts())})"
    )
    print("=" * 72)
    T.report()
    print()
    print(f"  ALL TESTS PASSED in {t_total:.2f}s")


if __name__ == "__main__":
    main()
