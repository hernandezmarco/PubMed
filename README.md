# PubMed AI Search

A personal research tool that turns plain-English questions into precise PubMed searches, ranks results by semantic relevance, and lets you chat with the literature using RAG (Retrieval-Augmented Generation).
It was developed as a small FLASK application for use on a Macbook-Pro laptop.  Hopefully, others will find it useful.

---

## What it does

Searching PubMed manually requires knowing MeSH terms, Boolean operators, and field tags. This tool removes that friction:

1. **Type a research question in plain English** — e.g. *"What is the role of gut microbiome in Parkinson's disease?"*
2. **Claude translates it** into an optimized PubMed query using MeSH terms and Boolean logic (extended thinking mode, so it reasons before answering).
3. **Optionally narrow by publication date** — an optional date-range picker restricts results to articles published between two dates; left blank, all dates are searched.
4. **NCBI returns up to 200 articles**; each abstract is embedded and ranked by cosine similarity to your original question.
5. **You review and select** the most relevant articles, set a relevance threshold with a slider, and save them as a named collection. Saving streams live progress — a PMC full-text fetch event per article, then a per-batch embedding progress event — so the progress bar advances smoothly even for large (up to 200-article) collections.
6. **The collection becomes a knowledge base** — full-text PMC articles are fetched where available, chunked, embedded, and stored in pgvector. The collection list shows each collection's article count split into "N full text" and "N abstract only" badges, so you can see the full-text/abstract-only mix without opening it.
7. **Chat with your collection** — ask follow-up questions and get cited, streamed answers grounded in the papers you saved. Inline citation markers like `[1]` are rendered as clickable superscript links that open the source article directly. Claude suggests four follow-up questions after every answer.
8. **Conversations are saved** — every chat thread is persisted to the database. A sidebar lists past conversations by title and date; click any entry to pick up where you left off, or start a fresh thread with the **+** button. Hover over a conversation to rename it inline with the pencil icon (✎) or delete it with the × button.
9. **Export a conversation** — click **Save ▾** in the chat toolbar to download the current conversation as a Word (`.docx`) or RTF (`.rtf`) file, including the question/answer history and metadata.
10. **Download collection metadata** — click **↓ CSV** at the top of the article list panel to download all articles in the collection as a CSV file (PMID, Journal, Title, Year, Authors, Abstract). The file is named after the collection and encoded as UTF-8 with BOM for Excel compatibility.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | Tested on 3.11; 3.12 may work |
| PostgreSQL 14+ with pgvector | `CREATE EXTENSION vector;` must be run in your database |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) — required |
| OpenAI API key | [platform.openai.com](https://platform.openai.com) — optional, adds OpenAI models to the chat dropdown |
| Ollama | [ollama.com](https://ollama.com) — optional, adds locally-run and/or cloud-hosted models to the chat dropdown |
| Docker (optional) | Only needed for container deployment |

### Install pgvector

```bash
# macOS (Homebrew)
brew install pgvector

# Ubuntu / Debian
sudo apt install postgresql-16-pgvector   # adjust version
```

Then enable it in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Local development setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd PubMed
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...

# Optional — enables OpenAI models in the chat model dropdown
OPENAI_API_KEY=sk-...

# Optional — local Ollama server for the chat model dropdown; defaults to
# http://localhost:11434 and is auto-detected (silently skipped if unreachable)
OLLAMA_BASE_URL=http://localhost:11434

# Optional — Ollama Cloud (https://ollama.com) key. Adds your account's hosted
# model catalog to the chat model dropdown, independent of OLLAMA_BASE_URL.
OLLAMA_API_KEY=your_ollama_cloud_api_key

DB_HOST=localhost
DB_NAME=pubmed_ai
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Required — every page requires login. Generate each with:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
# Two different values on purpose — different subsystems (JWT vs. Flask session/CSRF)
# shouldn't share a signing key.
JWT_SECRET_KEY=
FLASK_SECRET_KEY=

# SMTP2GO HTTP API key — sends the account-verification and password-reset emails.
# Registration requires clicking the verification link before the account can log
# in; unset just means the token is still created but the email never arrives
# (logged, not an error) — fine for local dev, but you'll need to verify the
# account directly in the DB to log in.
SMTP2GO_API=
EMAIL_FROM=PubMed AI <noreply@localhost>
APP_BASE_URL=http://127.0.0.1:8080

# Optional — NCBI E-utilities API key (raises rate limit 3 → 10 req/s)
# Register free at https://www.ncbi.nlm.nih.gov/account/
PUBMED_API=your_ncbi_api_key

# Optional — DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Optional — override the Claude models used for each task (defaults shown)
QUERY_BUILDER_MODEL=claude-opus-4-6
STARTER_QUESTIONS_MODEL=claude-opus-4-6
DEFAULT_CHAT_MODEL=claude-opus-4-6

# Optional — override the embedding model (changing this to a model with a
# different output dimension requires re-running scripts/migrate_embedding_dim.py)
EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings
```

See [Environment Variables](#environment-variables) below for the full list, including auth token lifetimes and rate limits.

### 4. Initialise the database

The app creates all tables automatically on first start. Just make sure the database exists:

```bash
createdb pubmed_ai
```

### 5. Run

```bash
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Every page requires an account — register one at `/register` and click the verification link emailed to you before you can log in, or run `python -m scripts.bootstrap_admin` to create/promote a specific account (auto-verified, skips the email step; it also reassigns any pre-auth collections, `user_id IS NULL`, to that account).

The first search will download the `NeuML/pubmedbert-base-embeddings` embedding model into `~/.cache/huggingface`. Subsequent starts are instant.

If you're upgrading an existing database from an older embedding model, run `python scripts/migrate_embedding_dim.py` once to resize `article_chunks.embedding` and re-embed existing chunks with the currently configured model.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key. |
| `OPENAI_API_KEY` | — | Optional. Enables OpenAI models in the chat model dropdown. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Optional. Local Ollama server; models pulled there are auto-discovered into the chat model dropdown. |
| `OLLAMA_API_KEY` | — | Optional. Ollama Cloud (`https://ollama.com`) key; adds your account's hosted model catalog to the chat model dropdown. |
| `DB_HOST` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | — | PostgreSQL connection. |
| `PUBMED_API` | — | Optional. NCBI E-utilities API key. Raises rate limit from 3 → 10 req/s. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `JWT_SECRET_KEY` | *(empty)* | **Required.** Signs/verifies access-token JWTs. Logs a startup warning if unset. |
| `JWT_ACCESS_TTL_MINUTES` | `30` | Access token lifetime. |
| `JWT_REFRESH_TTL_DAYS` | `30` | Refresh token lifetime (stored server-side, revocable). |
| `COOKIE_SECURE` | `true` | `Secure` flag on auth cookies. Only set `false` for plain-HTTP local dev without TLS. |
| `FLASK_SECRET_KEY` | *(empty)* | **Required.** Signs the Flask session cookie that CSRF tokens ride on — a different key from `JWT_SECRET_KEY` on purpose. Logs a startup warning if unset. |
| `LOGIN_RATE_LIMIT` | `5 per minute` | Rate limit on `POST /auth/login`. |
| `REGISTER_RATE_LIMIT` | `5 per hour` | Rate limit on `POST /auth/register`. |
| `FORGOT_PASSWORD_RATE_LIMIT` | `3 per hour` | Rate limit on `POST /auth/forgot-password`. |
| `RESEND_VERIFICATION_RATE_LIMIT` | `3 per hour` | Rate limit on `POST /auth/resend-verification`. |
| `PASSWORD_RESET_TTL_MINUTES` | `60` | How long a password-reset link stays valid. |
| `EMAIL_VERIFICATION_TTL_MINUTES` | `1440` | How long a new-account verification link stays valid (default 24h). |
| `SMTP2GO_API` | *(empty)* | API key for [SMTP2GO](https://www.smtp2go.com/)'s HTTP send API. Sends both verification and password-reset emails; unset just skips sending (the token is still created — logged, not an error), so accounts stay unverified and can't log in until verified directly in the DB. |
| `EMAIL_FROM` | `PubMed AI <noreply@localhost>` | From address on verification/reset emails. Must be a sender/domain verified in your SMTP2GO account. |
| `APP_BASE_URL` | `https://localhost:8080` | Base URL used to build the verification/reset links in emails. |

---

## Docker

The Docker image pre-bakes the embedding model so there is no download delay on startup. The container is served by **gunicorn over TLS**, not the Flask dev server.

### Generate a TLS certificate

Required once, **before building the image** — `certs/cert.pem` and `certs/key.pem` get baked into the image (`COPY . .`, not excluded by `.dockerignore`) so gunicorn's `--certfile`/`--keyfile` can find them without a runtime bind-mount:

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -nodes -keyout certs/key.pem -out certs/cert.pem -days 365 -subj "/CN=localhost"
```

This is self-signed, so browsers will show an untrusted-certificate warning unless you import `certs/cert.pem` into your OS/browser trust store — expected for a personal tool, not a bug. Rebuild the image any time you regenerate the cert. Because the private key is now part of the image, never push it to a registry or otherwise share it — `certs/` still stays out of git via `.gitignore`.

### Build

```bash
docker build -t pubmed-ai .
```

### Run

```bash
docker run -p 127.0.0.1:8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e OPENAI_API_KEY=sk-... \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e OLLAMA_API_KEY=your_ollama_cloud_api_key \
  -e DB_HOST=host.docker.internal \
  -e DB_NAME=pubmed_ai \
  -e DB_USER=your_db_user \
  -e DB_PASSWORD=your_db_password \
  -e JWT_SECRET_KEY=... \
  -e FLASK_SECRET_KEY=... \
  -e SMTP2GO_API=api-... \
  -e PUBMED_API=your_ncbi_api_key \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/logs:/app/logs \
  pubmed-ai
```

`JWT_SECRET_KEY` and `FLASK_SECRET_KEY` are required (generate each with `python -c "import secrets; print(secrets.token_urlsafe(48))"`) — use two different values, since they sign different things on purpose.

`OPENAI_API_KEY`, `OLLAMA_BASE_URL`, and `OLLAMA_API_KEY` are all optional — omit any of them to leave that provider out of the chat model dropdown. `OLLAMA_BASE_URL` (a local server) and `OLLAMA_API_KEY` (Ollama Cloud) are independent and can both be set at once; local Ollama runs on the host, so it needs `host.docker.internal` the same way `DB_HOST` does — Ollama Cloud does not, since it's a public endpoint.

`DB_HOST=host.docker.internal` connects to a PostgreSQL instance running on your Mac. Replace with a hostname or IP if your database is elsewhere.

> **Security note:** `-p 127.0.0.1:8080:8080` binds the published port to the host's loopback only — other machines on your LAN cannot reach the app. Using `-p 8080:8080` (without the IP prefix) would publish on all host interfaces. gunicorn itself must bind to `0.0.0.0` inside the container for Docker's port forwarding to work; this is safe because the container has its own isolated network namespace.

---

## Architecture

```
Browser
  │
  ▼
Flask (app.py)
  ├── GET/POST /                         Search page
  │     ├── Claude (Opus 4.6)       ──►  PubMed query (MeSH + Boolean, extended thinking)
  │     ├── NCBI esearch/efetch     ──►  Titles, authors, abstracts (optional pubdate range)
  │     └── sentence-transformers   ──►  Cosine similarity ranking
  │
  ├── POST /collections/save-stream      Save selected articles (SSE progress stream)
  │     ├── Phase 1 — parallel PMC fetches (as_completed), yields fetch event per article
  │     ├── Phase 2 — DB inserts (no event, fast)
  │     └── Phase 3 — chunk_text() (1 000-char chunks, 200-char overlap), then
  │                    embed in EMBED_BATCH_SIZE-sized batches, yielding an
  │                    embedding_progress event per batch (keeps gunicorn's sync
  │                    worker heartbeating on large collections) + pgvector
  │                    batch-insert (execute_values), then a done event
  │
  ├── POST /collections/<id>/ask         Chat with a collection (SSE)
  │     ├── sentence-transformers   ──►  Embed question
  │     ├── pgvector                ──►  Top-5 chunk retrieval (HNSW index)
  │     ├── Claude (selectable)     ──►  Streamed cited answer + 4 follow-up suggestions
  │     └── db.add_message()        ──►  Persist user + assistant messages
  │
  ├── GET  /api/models                           List available chat models (Anthropic/OpenAI + discovered Ollama local/cloud)
  ├── GET  /collections/<id>/starter-questions   Generate starter questions (cached, async)
  ├── GET  /collections/<id>/export.csv          Download article metadata as CSV
  ├── GET  /collections/<id>/conversations       List saved conversations
  ├── GET  /conversations/<id>/messages          Load a conversation's message history
  ├── PATCH /conversations/<id>/rename           Rename a conversation
  ├── POST /conversations/<id>/delete            Delete a conversation
  ├── POST /collections/<id>/delete              Delete a collection
  └── GET  /conversations/<id>/export?format=…  Download conversation as .docx or .rtf

Database (PostgreSQL + pgvector)
  ├── rag_collections        — id, name, user_query, pubmed_query, created_at
  ├── rag_articles           — id, collection_id, pmid, title, authors, journal, year,
  │                            abstract, url, has_full_text, pmcid
  ├── article_chunks         — id, pmid, chunk_index, text, embedding vector(768)
  │                            UNIQUE(pmid, chunk_index) · HNSW index on embedding
  ├── conversations          — id, collection_id, title (first question), created_at
  └── conversation_messages  — id, conversation_id, role (user|assistant), content,
                               citations JSONB (num/pmid/title/url per cited article), created_at
```

### Embedding model

`NeuML/pubmedbert-base-embeddings` (768-dimensional vectors) via [sentence-transformers](https://www.sbert.net/) — PubMedBERT fine-tuned on PubMed title/abstract pairs, chosen for biomedical domain relevance over general-purpose embedding models. Used for both indexing article chunks and embedding queries at search/ask time. The model is cached in `~/.cache/huggingface` and pre-baked into the Docker image. Switching to a model with a different output dimension requires running `scripts/migrate_embedding_dim.py` to resize `article_chunks.embedding` and re-embed existing chunks.

### Chat models

All Claude calls go through [litellm](https://github.com/BerriAI/litellm), which routes by model-ID prefix to Anthropic, OpenAI, a local Ollama server, or Ollama Cloud — no code changes needed to add a provider, only credentials/config.

| Task | Default model | Override env var |
|---|---|---|
| PubMed query generation (extended thinking) | `claude-opus-4-6` | `QUERY_BUILDER_MODEL` |
| Starter question generation | `claude-opus-4-6` | `STARTER_QUESTIONS_MODEL` |
| Collection chat | `claude-opus-4-6` | `DEFAULT_CHAT_MODEL` |

> Query generation and starter questions make a single non-streaming call, so they only support Claude. Ollama Cloud rejects non-streaming chat requests (401 Unauthorized) — only the collection-chat path streams, so it's the only place Ollama (local or cloud) models are usable.

The collection-chat model can also be switched at runtime from the collection page. The dropdown is populated live from `GET /api/models`, which lists:
- Anthropic/OpenAI models from `cfg.CHAT_MODELS` (config.py) whose `requires_key` env var is set,
- any models currently pulled in a locally reachable Ollama server (`OLLAMA_BASE_URL`, default `http://localhost:11434`), discovered via `/api/tags` on every request — so `ollama pull <model>` shows up without restarting the app, and
- if `OLLAMA_API_KEY` is set, your Ollama Cloud account's hosted model catalog (`https://ollama.com/api/tags`, bearer-authenticated) — a fixed list of models available to your account, not something you "pull." Labeled "Ollama (cloud)" in the dropdown to distinguish it from the local server.

The selected model preference is persisted in `localStorage`. Ollama models (local and cloud) are always shown as `$0.00` in the usage footer — local genuinely is free, but Ollama Cloud usage is actually metered by Ollama on their end; this app just doesn't track/display that cost.

All tunable values — model IDs, system prompts, timeouts, chunk sizes, token limits — live in `config.py`.

### Logging

Logs go to the console and `logs/app.log` (rotating, 10 MB × 5 backups). Verbosity is controlled by the `LOG_LEVEL` env var.

---

## Project structure

```
app.py                   # Flask routes and all business logic
auth.py                  # Password hashing, JWT, login_required(_page), refresh/reset tokens, reset email
config.py                # Central configuration — models, prompts, timeouts, limits, auth settings
db.py                    # PostgreSQL / pgvector layer
wsgi.py                  # gunicorn entrypoint (runs db.init_db(), exposes `application`)
requirements.txt
Dockerfile
.dockerignore

scripts/
  bootstrap_admin.py         # One-time: create/promote an admin user, reassign pre-auth collections
  migrate_embedding_dim.py   # Re-runnable: resize article_chunks.embedding and re-embed chunks
                             # after changing EMBEDDING_MODEL to a model with a different dimension

templates/
  base.html              # Shared layout (progress bar, CSRF meta tag, account bar, breadcrumbs, nav)
  index.html             # Search + relevance review panel
  collections.html       # Collection list
  collection.html        # Collection detail + 3-column layout (articles | conversations | chat)
  login.html / register.html
  forgot_password.html / reset_password.html

static/
  base.css / base.js     # Shared styles, progress bar, authFetch() CSRF+silent-refresh wrapper, logout
  auth.css / auth.js     # Login/register/forgot/reset forms
  index.css / index.js   # Search page and review panel
  collections.css / collections.js
  collection.css / collection.js   # Chat UI, conversation sidebar, history loader

tests/
  test_app.py            # Unit tests — routes, pure functions, helpers (pytest)
  test_auth.py           # Unit tests — auth.py, /auth/* routes
  test_db.py             # Unit tests — database layer (mocked psycopg2)

logs/
  .gitkeep               # Directory tracked; *.log is git-ignored
```

---

## Contributing

Bug reports and ideas are welcome.

1. Fork the repository and create a feature branch from `main`.
2. Make your changes — keep commits focused and descriptive.
3. Run the unit tests and smoke-test locally before opening a pull request:

```bash
source .venv/bin/activate
pytest tests/
python app.py
```

4. Open a PR against `main` with a clear description of what changed and why.

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.