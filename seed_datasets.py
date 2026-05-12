"""One-shot seeding script for the hackathon datasets.

Creates (or overwrites) two private datasets on the Hub with dummy rows so the
Space has something to read on first launch. Re-run this whenever the schema
in ``data.py`` changes.

Usage:
    HF_TOKEN=hf_... python seed_datasets.py
"""

from __future__ import annotations

import os

from datasets import Dataset

from data import (
    EMPTY_VALIDATION,
    EMPTY_VOTE,
    PARTICIPANTS_FEATURES,
    PARTICIPANTS_REPO,
    PROMPTS_FEATURES,
    PROMPTS_REPO,
)

DUMMY_PARTICIPANTS = [
    {
        "username": "mariagrandury",
        "language": "es",
        "country": "es",
        "gmail": "maria.grandury@gmail.com",
    },
    {
        "username": "alice-cl",
        "language": "es",
        "country": "cl",
        "gmail": "alice.cl@gmail.com",
    },
    {
        "username": "bruno-br",
        "language": "pt",
        "country": "br",
        "gmail": "bruno.br@gmail.com",
    },
    {
        "username": "carla-co",
        "language": "es",
        "country": "co",
        "gmail": "carla.co@gmail.com",
    },
    {
        "username": "diogo-pt",
        "language": "pt",
        "country": "pt",
        "gmail": "diogo.pt@gmail.com",
    },
    {
        "username": "evan-en",
        "language": "en",
        "country": "es",
        "gmail": "evan.en@gmail.com",
    },
]


def _validation(username: str, choice: str = "relevant") -> dict:
    return {"choice": choice, "username": username}


def _vote(username: str, choice: str) -> dict:
    return {"choice": choice, "username": username}


# Mix of unvalidated, partially validated, and fully-validated prompts so that
# every tab in the app has something to display on first launch.
DUMMY_PROMPTS = [
    {
        "username": "mariagrandury",
        "language": "es",
        "country": "es",
        "system_prompt": "",
        "prompt": "¿Qué se suele cenar en Nochevieja en España?",
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
    },
    {
        "username": "alice-cl",
        "language": "es",
        "country": "cl",
        "system_prompt": "",
        "prompt": "¿Cuál es la diferencia entre 'once' y 'cena' en Chile?",
        "prompt_validation_1": _validation("bruno-br"),
        "prompt_validation_2": dict(EMPTY_VALIDATION),
        "prompt_validation_3": dict(EMPTY_VALIDATION),
        "answer_a": "",
        "model_a": "",
        "answer_b": "",
        "model_b": "",
        "answer_chosen_1": dict(EMPTY_VOTE),
        "answer_chosen_2": dict(EMPTY_VOTE),
        "answer_chosen_3": dict(EMPTY_VOTE),
    },
    {
        "username": "bruno-br",
        "language": "pt",
        "country": "br",
        "system_prompt": "",
        "prompt": "Qual é o prato típico do São João no Nordeste do Brasil?",
        "prompt_validation_1": _validation("alice-cl"),
        "prompt_validation_2": _validation("carla-co"),
        "prompt_validation_3": _validation("diogo-pt"),
        "answer_a": (
            "No Nordeste, o São João é celebrado com canjica, pamonha, "
            "bolo de milho e quentão."
        ),
        "model_a": "qwen/qwen3.5",
        "answer_b": (
            "O prato típico mais associado ao São João é a paçoca de "
            "amendoim, embora também haja milho cozido."
        ),
        "model_b": "meta-llama/llama-3.1-70b",
        "answer_chosen_1": _vote("alice-cl", "a"),
        "answer_chosen_2": dict(EMPTY_VOTE),
        "answer_chosen_3": dict(EMPTY_VOTE),
    },
    {
        "username": "carla-co",
        "language": "es",
        "country": "co",
        "system_prompt": "",
        "prompt": "¿Qué se come tradicionalmente en una novena de Navidad en Colombia?",
        "prompt_validation_1": _validation("mariagrandury"),
        "prompt_validation_2": _validation("bruno-br"),
        "prompt_validation_3": _validation("diogo-pt"),
        "answer_a": (
            "Buñuelos, natilla y manjar blanco son los infaltables; "
            "se acompañan con chocolate caliente."
        ),
        "model_a": "qwen/qwen3.5",
        "answer_b": (
            "Lo más común es comer pan de yuca y arepas, junto con "
            "una taza de café tinto."
        ),
        "model_b": "meta-llama/llama-3.1-70b",
        "answer_chosen_1": dict(EMPTY_VOTE),
        "answer_chosen_2": dict(EMPTY_VOTE),
        "answer_chosen_3": dict(EMPTY_VOTE),
    },
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Export a token with write access to the "
            "private repos before running this script."
        )

    participants = Dataset.from_list(DUMMY_PARTICIPANTS, features=PARTICIPANTS_FEATURES)
    participants.push_to_hub(PARTICIPANTS_REPO, private=True, token=token)
    print(f"Pushed {len(participants)} rows to {PARTICIPANTS_REPO}")

    prompts = Dataset.from_list(DUMMY_PROMPTS, features=PROMPTS_FEATURES)
    prompts.push_to_hub(PROMPTS_REPO, private=True, token=token)
    print(f"Pushed {len(prompts)} rows to {PROMPTS_REPO}")


if __name__ == "__main__":
    main()
