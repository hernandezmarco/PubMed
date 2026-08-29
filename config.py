# Copyright (C) 2025 Marco Hernandez <ragettyandy@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# For information contact Marco Hernandez <ragettyandy@gmail.com>

"""
Central configuration for the PubMed AI application.

Numeric and model values are plain constants — change them here to affect the
whole application.  Model IDs can also be overridden via environment variables
so that different deployments can swap models without a code change.
"""

import os

# ── Provider credentials ──────────────────────────────────────────────────────

# Env var names, also referenced by CHAT_MODELS' "requires_key" entries below.
_ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
_OPENAI_API_KEY_ENV    = "OPENAI_API_KEY"

ANTHROPIC_API_KEY = os.getenv(_ANTHROPIC_API_KEY_ENV, "")
OPENAI_API_KEY    = os.getenv(_OPENAI_API_KEY_ENV,    "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_API_KEY    = os.getenv("OLLAMA_API_KEY",    "")

# Fixed — not user-configurable like OLLAMA_BASE_URL. Ollama Cloud requires
# streaming chat requests (non-streaming calls return 401 Unauthorized).
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

# ── Auth ──────────────────────────────────────────────────────────────────────

JWT_SECRET_KEY          = os.getenv("JWT_SECRET_KEY", "")
JWT_ACCESS_TTL_MINUTES  = int(os.getenv("JWT_ACCESS_TTL_MINUTES", "30"))
JWT_REFRESH_TTL_DAYS    = int(os.getenv("JWT_REFRESH_TTL_DAYS", "30"))
JWT_ALGORITHM           = "HS256"

# Whether auth cookies get the Secure flag (browser will refuse to send them over plain
# HTTP if set). Defaults on — this app is meant to be served over TLS (see Dockerfile).
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() != "false"

# Signs the Flask session cookie that flask-wtf's CSRF token rides on — distinct from
# JWT_SECRET_KEY on purpose (different subsystems shouldn't share a signing key).
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")

# flask-limiter uses in-memory (per-worker) storage — see README for the tradeoff.
LOGIN_RATE_LIMIT                = os.getenv("LOGIN_RATE_LIMIT",                "5 per minute")
REGISTER_RATE_LIMIT             = os.getenv("REGISTER_RATE_LIMIT",             "5 per hour")
FORGOT_PASSWORD_RATE_LIMIT      = os.getenv("FORGOT_PASSWORD_RATE_LIMIT",      "3 per hour")  # NOSONAR
RESEND_VERIFICATION_RATE_LIMIT  = os.getenv("RESEND_VERIFICATION_RATE_LIMIT",  "3 per hour")

PASSWORD_RESET_TTL_MINUTES      = int(os.getenv("PASSWORD_RESET_TTL_MINUTES",      "60"))  # NOSONAR
EMAIL_VERIFICATION_TTL_MINUTES  = int(os.getenv("EMAIL_VERIFICATION_TTL_MINUTES",  "1440"))

# ── Email (SMTP2GO HTTP API — https://api.smtp2go.com/v3/email/send) ──────────
# Registration requires clicking the emailed link before the account can log in;
# SMTP2GO_API unset just means the account is created but the email never
# arrives (logged, not an error) — fine for local dev, sign in blocked until an
# admin verifies the row directly.

SMTP2GO_API = os.getenv("SMTP2GO_API", "")
EMAIL_FROM  = os.getenv("EMAIL_FROM", "PubMed AI <noreply@localhost>")

# Base URL used to build the verification / password-reset links in emails
# (e.g. https://host:8080). No trailing slash.
APP_BASE_URL = os.getenv("APP_BASE_URL", "https://localhost:8080")

# ── Models ────────────────────────────────────────────────────────────────────

# Also used as the CHAT_MODELS key for this model below.
_DEFAULT_MODEL_ID = "claude-opus-4-6"

QUERY_BUILDER_MODEL     = os.getenv("QUERY_BUILDER_MODEL",     _DEFAULT_MODEL_ID)
STARTER_QUESTIONS_MODEL = os.getenv("STARTER_QUESTIONS_MODEL", _DEFAULT_MODEL_ID)
DEFAULT_CHAT_MODEL      = os.getenv("DEFAULT_CHAT_MODEL",      _DEFAULT_MODEL_ID)

# NeuML/pubmedbert-base-embeddings — PubMedBERT fine-tuned for biomedical semantic
# search, trained specifically on PubMed title/abstract pairs. Only ships
# PyTorch/safetensors weights (no ONNX), so it's loaded via sentence-transformers
# rather than fastembed. 768-dim — EMBEDDING_DIM below must match whatever model
# is configured here, since it's baked into the article_chunks.embedding column type.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "768"))

# All chat models available for selection. Entries whose requires_key env var is
# not set at startup are hidden from the UI. Ollama models are appended at
# runtime via dynamic discovery (see app.py:available_chat_models).
CHAT_MODELS: dict[str, dict] = {
    _DEFAULT_MODEL_ID: {
        "display":      "Opus 4.6",
        "provider":     "anthropic",
        "requires_key": _ANTHROPIC_API_KEY_ENV,
        "input_price":  15.00,
        "output_price": 75.00,
    },
    "claude-sonnet-4-6": {
        "display":      "Sonnet 4.6",
        "provider":     "anthropic",
        "requires_key": _ANTHROPIC_API_KEY_ENV,
        "input_price":   3.00,
        "output_price": 15.00,
    },
    "claude-haiku-4-5-20251001": {
        "display":      "Haiku 4.5",
        "provider":     "anthropic",
        "requires_key": _ANTHROPIC_API_KEY_ENV,
        "input_price":   0.80,
        "output_price":  4.00,
    },
    "gpt-4o": {
        "display":      "GPT-4o",
        "provider":     "openai",
        "requires_key": _OPENAI_API_KEY_ENV,
        "input_price":   2.50,
        "output_price": 10.00,
    },
    "o3": {
        "display":      "o3",
        "provider":     "openai",
        "requires_key": _OPENAI_API_KEY_ENV,
        "input_price":  10.00,
        "output_price": 40.00,
    },
}

# ── Generation limits (tokens) ────────────────────────────────────────────────

MAX_TOKENS_PUBMED_QUERY = 1536   # 1024 budget_tokens for thinking + ~512 for output
MAX_TOKENS_STARTER_QS   = 256
MAX_TOKENS_RAG_RESPONSE = 4096

# Anthropic requires thinking.budget_tokens >= 1024 and < max_tokens.
PUBMED_QUERY_THINKING_BUDGET = 1024

# ── NCBI / PubMed ─────────────────────────────────────────────────────────────

PUBMED_BASE      = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY   = os.getenv("PUBMED_API", "")   # raises rate limit 3 → 10 req/s when set
NCBI_BACKOFF_MAX = 30.0   # seconds; cap on exponential back-off delay

# ── Request timeouts (seconds) ────────────────────────────────────────────────

TIMEOUT_ESEARCH      = 10
TIMEOUT_EFETCH       = 15
TIMEOUT_ELINK        = 30
TIMEOUT_PMC_FULLTEXT = 30
ELINK_BATCH_SIZE     = 50

# ── Search / retrieval ────────────────────────────────────────────────────────

MAX_RESULTS_DEFAULT = 25
MAX_RESULTS_MIN     = 25
MAX_RESULTS_MAX     = 200
SEMANTIC_SEARCH_K   = 5   # top-k chunks retrieved per RAG query

# ── Embedding ────────────────────────────────────────────────────────────────
# EMBED_THREADS caps PyTorch's intra-op CPU thread pool — the primary lever for
# preventing CPU spikes in memory-constrained containers. Set to 1 to keep
# CPU usage flat; raise to 2-4 if the host has spare cores.
# EMBED_BATCH_SIZE controls texts per encode() forward pass. Lowering it reduces
# peak memory and smooths CPU load at the cost of slightly longer total
# embedding time.

EMBED_THREADS    = int(os.environ.get("EMBED_THREADS",    "1"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# ── Text chunking ─────────────────────────────────────────────────────────────

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Display ───────────────────────────────────────────────────────────────────

AUTHORS_DISPLAY_MAX = 6   # show up to N authors, then "et al."

# ── Static asset version (bump to bust browser cache after JS/CSS changes) ────

STATIC_VERSION = "5"

# ── Streaming ─────────────────────────────────────────────────────────────────

RAG_DELIMITER = "\n===\n"   # separates the answer from the follow-up JSON array

# ── System prompts ────────────────────────────────────────────────────────────

PROMPT_PUBMED_QUERY = (
    "You are a biomedical librarian expert in PubMed search syntax. "
    "Convert the user's plain-English research question into an optimal PubMed "
    "query using MeSH terms, field tags ([MeSH Terms], [Title/Abstract], etc.), "
    "and Boolean operators (AND, OR, NOT). "
    "Return ONLY the raw query string — no explanation, no markdown, no quotes."
)

PROMPT_STARTER_QUESTIONS = (
    "Generate exactly 4 concise research questions a user might want to ask about "
    "this collection. Return ONLY a JSON array of 4 strings, no other text."
)

PROMPT_RAG_SYSTEM = (
    "You are a biomedical research assistant. Answer the user's question using "
    "ONLY the numbered article excerpts provided. Cite sources inline as [1], [2], etc. "
    "Be as thorough and detailed as possible: explain mechanisms, report specific findings, "
    "numbers, and comparisons from the excerpts, and synthesize across multiple sources rather "
    "than giving a brief summary. If the excerpts lack sufficient information, say so.\n\n"
    "After your answer output a line containing only === followed immediately by a "
    "JSON array of exactly 4 concise follow-up questions the user might ask next, "
    "based on your answer. Example:\n"
    "===\n"
    "[\"Question one?\", \"Question two?\", \"Question three?\", \"Question four?\"]"
)