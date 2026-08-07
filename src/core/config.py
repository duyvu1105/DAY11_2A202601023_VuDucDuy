"""
Lab 11 — Configuration & API Key Setup
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv optional; env can be set by the shell instead
    load_dotenv = None


_REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 so Unicode output survives Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_env() -> None:
    """Load ``.env`` from the repo root and map legacy Vertex env names.

    The Google GenAI SDK reads ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_LOCATION``
    while the lab ``.env`` ships the legacy ``VERTEXAI_PROJECT`` /
    ``VERTEXAI_LOCATION`` names.  We map them so ``genai.Client()`` and ADK pick
    up the right project/location automatically.
    """
    env_path = os.environ.get("DAY11_ENV_FILE") or (_REPO_ROOT / ".env")
    if load_dotenv is not None and env_path and Path(env_path).exists():
        load_dotenv(Path(env_path), override=False)

    if os.environ.get("VERTEXAI_PROJECT") and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ["VERTEXAI_PROJECT"]
    if os.environ.get("VERTEXAI_LOCATION") and not os.environ.get("GOOGLE_CLOUD_LOCATION"):
        os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ["VERTEXAI_LOCATION"]


def get_model() -> str:
    """Return the plain model id configured in ``.env``.

    ``MODEL=vertex_ai/gemini-3.6-flash`` is a provider-qualified name.  The
    google-genai / ADK SDKs already know which backend to use from
    ``GOOGLE_GENAI_USE_VERTEXAI`` + project/location env vars, so we strip the
    ``vertex_ai/`` prefix and hand the SDK the bare model id.
    """
    model = os.environ.get("MODEL", "").strip() or "gemini-3.6-flash"
    if "/" in model and model.split("/", 1)[0] in {
        "vertex_ai",
        "google_ai",
        "google",
        "gemini",
    }:
        model = model.split("/", 1)[1]
    return model


def setup_api_key():
    """Load ``.env`` and report which backend the SDK will use.

    Vertex AI mode authenticates through Application Default Credentials (or
    ``GOOGLE_APPLICATION_CREDENTIALS``); Gemini API mode needs ``GOOGLE_API_KEY``.
    We intentionally keep ``GOOGLE_GENAI_USE_VERTEXAI`` from ``.env`` instead of
    forcing the old default (0) so the lab runs on Vertex AI.
    """
    load_env()
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if not use_vertex and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")
    print(
        "Environment loaded "
        f"(backend={'Vertex AI' if use_vertex else 'Gemini API'}, "
        f"model={MODEL}, project={os.environ.get('VERTEXAI_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT') or '-'})."
    )


# Load .env as soon as the module is imported so every consumer (agents,
# guardrails, judge, attacks) sees the Vertex AI config + model id.
load_env()
ensure_utf8_stdio()
MODEL = get_model()


# Allowed banking topics (used by topic_filter)
ALLOWED_TOPICS = [
    "banking", "account", "transaction", "transfer",
    "loan", "interest", "savings", "credit",
    "deposit", "withdrawal", "balance", "payment",
    "tai khoan", "giao dich", "tiet kiem", "lai suat",
    "chuyen tien", "the tin dung", "so du", "vay",
    "ngan hang", "atm",
]

# Blocked topics (immediate reject)
BLOCKED_TOPICS = [
    "hack", "exploit", "weapon", "drug", "illegal",
    "violence", "gambling", "bomb", "kill", "steal",
]
