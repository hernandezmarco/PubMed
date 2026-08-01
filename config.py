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

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")

# ── Models ────────────────────────────────────────────────────────────────────

QUERY_BUILDER_MODEL     = os.getenv("QUERY_BUILDER_MODEL",     "claude-opus-4-6")
STARTER_QUESTIONS_MODEL = os.getenv("STARTER_QUESTIONS_MODEL", "claude-opus-4-6")
DEFAULT_CHAT_MODEL      = os.getenv("DEFAULT_CHAT_MODEL",      "claude-opus-4-6")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# All chat models available for selection. Entries whose requires_key env var is
# not set at startup are hidden from the UI. Ollama models are appended at
# runtime via dynamic discovery (see app.py:available_chat_models).
CHAT_MODELS: dict[str, dict] = {
    "claude-opus-4-6": {
        "display":      "Opus 4.6",
        "provider":     "anthropic",
        "requires_key": "ANTHROPIC_API_KEY",
        "input_price":  15.00,
        "output_price": 75.00,
    },
    "claude-sonnet-4-6": {
        "display":      "Sonnet 4.6",
        "provider":     "anthropic",
        "requires_key": "ANTHROPIC_API_KEY",
        "input_price":   3.00,
        "output_price": 15.00,
    },
    "claude-haiku-4-5-20251001": {
        "display":      "Haiku 4.5",
        "provider":     "anthropic",
        "requires_key": "ANTHROPIC_API_KEY",
        "input_price":   0.80,
        "output_price":  4.00,
    },
    "gpt-4o": {
        "display":      "GPT-4o",
        "provider":     "openai",
        "requires_key": "OPENAI_API_KEY",
        "input_price":   2.50,
        "output_price": 10.00,
    },
    "o3": {
        "display":      "o3",
        "provider":     "openai",
        "requires_key": "OPENAI_API_KEY",
        "input_price":  10.00,
        "output_price": 40.00,
    },
}

# ── Generation limits (tokens) ────────────────────────────────────────────────

MAX_TOKENS_PUBMED_QUERY = 1536   # 512 budget_tokens for thinking + ~1k for output
MAX_TOKENS_STARTER_QS   = 256
MAX_TOKENS_RAG_RESPONSE = 2560

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
# EMBED_THREADS caps the ONNX Runtime thread pool — the primary lever for
# preventing CPU spikes in memory-constrained containers. Set to 1 to keep
# CPU usage flat; raise to 2-4 if the host has spare cores.
# EMBED_BATCH_SIZE controls texts per ONNX forward pass (default upstream: 256).
# Lowering it reduces peak memory and smooths CPU load at the cost of
# slightly longer total embedding time.

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
    "Be concise and precise. If the excerpts lack sufficient information, say so.\n\n"
    "After your answer output a line containing only === followed immediately by a "
    "JSON array of exactly 4 concise follow-up questions the user might ask next, "
    "based on your answer. Example:\n"
    "===\n"
    "[\"Question one?\", \"Question two?\", \"Question three?\", \"Question four?\"]"
)