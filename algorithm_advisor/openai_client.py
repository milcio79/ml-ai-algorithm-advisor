from __future__ import annotations

import os

import httpx
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from models import AlgorithmRecommendation, ProjectProfile
from prompts import build_ranking_prompt


DEFAULT_MODEL = "gpt-5.5"


class MissingOpenAIKeyError(RuntimeError):
    pass


def load_environment() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)
    load_dotenv(override=False)


def get_openai_model() -> str:
    load_environment()
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def rank_with_chatgpt(
    project_profile: ProjectProfile,
    candidate_algorithms: list[AlgorithmRecommendation],
) -> str:
    load_environment()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingOpenAIKeyError(
            "OPENAI_API_KEY is missing. Local recommendations still work, but ChatGPT ranking requires an API key."
        )

    model = get_openai_model()
    # The local Codex/VSCode environment may define placeholder proxy variables.
    # trust_env=False prevents httpx from routing OpenAI traffic through a broken local proxy.
    http_client = httpx.Client(trust_env=False, timeout=120)
    client = OpenAI(api_key=api_key, http_client=http_client)
    prompt = build_ranking_prompt(project_profile, candidate_algorithms)

    response = client.responses.create(
        model=model,
        input=prompt,
    )
    return response.output_text
