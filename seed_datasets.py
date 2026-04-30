"""One-shot seeding script for the hackathon datasets.

Creates (or overwrites) two private datasets on the Hub with dummy rows so the
Space has something to read on first launch.

Usage:
    HF_TOKEN=hf_... python seed_datasets.py
"""

from __future__ import annotations

import os

from datasets import Dataset, Features, Value

PARTICIPANTS_REPO = "mariagrandury/hackathon_participants"
PROMPTS_REPO = "mariagrandury/cultural_preferences"


PARTICIPANTS_FEATURES = Features(
    {
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "gmail": Value("string"),
    }
)

VALIDATION_STRUCT = {
    "validated": Value("bool"),
    "username": Value("string"),
}

PROMPTS_FEATURES = Features(
    {
        "username": Value("string"),
        "language": Value("string"),
        "country": Value("string"),
        "prompt": Value("string"),
        "prompt_validation_1": VALIDATION_STRUCT,
        "prompt_validation_2": VALIDATION_STRUCT,
        "prompt_validation_3": VALIDATION_STRUCT,
        "answer_a": Value("string"),
        "model_a": Value("string"),
        "answer_b": Value("string"),
        "model_b": Value("string"),
    }
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


def _empty_validation() -> dict:
    return {"validated": False, "username": ""}


DUMMY_PROMPTS = [
    {
        "username": "mariagrandury",
        "language": "es",
        "country": "es",
        "prompt": "¿Qué se suele cenar en Nochevieja en España?",
        "prompt_validation_1": _empty_validation(),
        "prompt_validation_2": _empty_validation(),
        "prompt_validation_3": _empty_validation(),
        "answer_a": "",
        "model_a": "",
        "answer_b": "",
        "model_b": "",
    },
    {
        "username": "bruno-br",
        "language": "pt",
        "country": "br",
        "prompt": "Qual é o prato típico do São João no Nordeste do Brasil?",
        "prompt_validation_1": _empty_validation(),
        "prompt_validation_2": _empty_validation(),
        "prompt_validation_3": _empty_validation(),
        "answer_a": "",
        "model_a": "",
        "answer_b": "",
        "model_b": "",
    },
    {
        "username": "alice-cl",
        "language": "es",
        "country": "cl",
        "prompt": "¿Cuál es la diferencia entre 'once' y 'cena' en Chile?",
        "prompt_validation_1": _empty_validation(),
        "prompt_validation_2": _empty_validation(),
        "prompt_validation_3": _empty_validation(),
        "answer_a": "",
        "model_a": "",
        "answer_b": "",
        "model_b": "",
    },
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Export a token with write access to the "
            "private repos before running this script."
        )

    participants = Dataset.from_list(
        DUMMY_PARTICIPANTS, features=PARTICIPANTS_FEATURES
    )
    participants.push_to_hub(PARTICIPANTS_REPO, private=True, token=token)
    print(f"Pushed {len(participants)} rows to {PARTICIPANTS_REPO}")

    prompts = Dataset.from_list(DUMMY_PROMPTS, features=PROMPTS_FEATURES)
    prompts.push_to_hub(PROMPTS_REPO, private=True, token=token)
    print(f"Pushed {len(prompts)} rows to {PROMPTS_REPO}")


if __name__ == "__main__":
    main()
