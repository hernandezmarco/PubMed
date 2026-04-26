# PubMed AI Search

A personal research tool that turns plain-English questions into precise PubMed searches, ranks results by semantic relevance, and lets you chat with the literature using RAG (Retrieval-Augmented Generation).
It was developed as a small FLASK application for use on a Macbook-Pro laptop.  Hopefully, others will find it useful.

---

## What it does

Searching PubMed manually requires knowing MeSH terms, Boolean operators, and field tags. This tool removes that friction:

1. **Type a research question in plain English** — e.g. *"What is the role of gut microbiome in Parkinson's disease?"*
2. **Claude translates it** into an optimized PubMed query using MeSH terms and Boolean logic (extended thinking mode, so it reasons before answering).
3. **NCBI returns up to 200 articles**; each abstract is embedded and ranked by cosine similarity to your original question.
4. **You review and select** the most relevant articles, set a relevance threshold with a slider, and save them as a named collection.
5. **The collection becomes a knowledge base** — full-text PMC articles are fetched where available, chunked, embedded, and stored in pgvector.
6. **Chat with your collection** — ask follow-up questions and get cited, streamed answers grounded in the papers you saved. Inline citation markers like `[1]` are rendered as clickable superscript links that open the source article directly. Claude suggests four follow-up questions after every answer.
7. **Conversations are saved** — every chat thread is persisted to the database. A sidebar lists past conversations by title and date; click any entry to pick up where you left off, or start a fresh thread with the **+** button. Hover over a conversation to rename it inline with the pencil icon (✎) or delete it with the × button.
8. **Export a conversation** — click **Save ▾** in the chat toolbar to download the current conversation as a Word (`.docx`) or RTF (`.rtf`) file, including the question/answer history and metadata.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | Tested on 3.11; 3.12 may work |
| PostgreSQL 14+ with pgvector | `CREATE EXTENSION vector;` must be run in your database |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) |
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

DB_HOST=localhost
DB_NAME=pubmed_ai
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# Optional — NCBI E-utilities API key (raises rate limit 3 → 10 req/s)
# Register free at https://www.ncbi.nlm.nih.gov/account/
PUBMED_API=your_ncbi_api_key

# Optional — DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Optional — override the Claude models used for each task (defaults shown)
QUERY_BUILDER_MODEL=claude-opus-4-6
STARTER_QUESTIONS_MODEL=claude-opus-4-6
DEFAULT_CHAT_MODEL=claude-opus-4-6

# Optional — override the embedding model
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### 4. Initialise the database

The app creates all tables automatically on first start. Just make sure the database exists:

```bash
createdb pubmed_ai
```

### 5. Run

```bash
python app.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080).

The first search will download the `BAAI/bge-small-en-v1.5` embedding model (~130 MB) into `~/.cache/fastembed`. Subsequent starts are instant.

---

## Docker

The Docker image pre-bakes the embedding model so there is no download delay on startup.

### Build

```bash
docker build -t pubmed-ai .
```

### Run

```bash
docker run -p 127.0.0.1:8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e DB_HOST=host.docker.internal \
  -e DB_NAME=pubmed_ai \
  -e DB_USER=your_db_user \
  -e DB_PASSWORD=your_db_password \
  -e PUBMED_API=your_ncbi_api_key \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/logs:/app/logs \
  pubmed-ai
```

`DB_HOST=host.docker.internal` connects to a PostgreSQL instance running on your Mac. Replace with a hostname or IP if your database is elsewhere.

> **Security note:** `-p 127.0.0.1:8080:8080` binds the published port to the host's loopback only — other machines on your LAN cannot reach the app. Using `-p 8080:8080` (without the IP prefix) would publish on all host interfaces. Flask itself must bind to `0.0.0.0` inside the container for Docker's port forwarding to work; this is safe because the container has its own isolated network namespace.

---

## Architecture

```
Browser
  │
  ▼
Flask (app.py)
  ├── GET/POST /                         Search page
  │     ├── Claude (Opus 4.6)       ──►  PubMed query (MeSH + Boolean, extended thinking)
  │     ├── NCBI esearch/efetch     ──►  Titles, authors, abstracts
  │     └── fastembed               ──►  Cosine similarity ranking
  │
  ├── POST /collections                  Save selected articles (synchronous)
  │     ├── NCBI elink/efetch       ──►  PMC full text
  │     ├── chunk_text()            ──►  1 000-char chunks, 200-char overlap
  │     ├── fastembed               ──►  Batch-embed all chunks in one call
  │     └── pgvector                ──►  Batch-insert embeddings (execute_values)
  │
  ├── POST /collections/save-stream      Save selected articles (SSE progress stream)
  │     ├── Phase 1 — parallel PMC fetches (as_completed), yields fetch event per article
  │     ├── Phase 2 — DB inserts (no event, fast)
  │     └── Phase 3 — batch embed + pgvector insert, yields embedding then done events
  │
  ├── POST /collections/<id>/ask         Chat with a collection (SSE)
  │     ├── fastembed               ──►  Embed question
  │     ├── pgvector                ──►  Top-5 chunk retrieval (HNSW index)
  │     ├── Claude (selectable)     ──►  Streamed cited answer + 4 follow-up suggestions
  │     └── db.add_message()        ──►  Persist user + assistant messages
  │
  ├── GET  /collections/<id>/starter-questions   Generate starter questions (cached, async)
  ├── GET  /collections/<id>/conversations       List saved conversations
  ├── GET  /conversations/<id>/messages          Load a conversation's message history
  ├── PATCH /conversations/<id>/rename           Rename a conversation
  ├── POST /conversations/<id>/delete            Delete a conversation
  └── GET  /conversations/<id>/export?format=…  Download conversation as .docx or .rtf

Database (PostgreSQL + pgvector)
  ├── rag_collections        — id, name, user_query, pubmed_query, created_at
  ├── rag_articles           — id, collection_id, pmid, title, authors, journal, year,
  │                            abstract, url, has_full_text, pmcid
  ├── article_chunks         — id, pmid, chunk_index, text, embedding vector(384)
  │                            UNIQUE(pmid, chunk_index) · HNSW index on embedding
  ├── conversations          — id, collection_id, title (first question), created_at
  └── conversation_messages  — id, conversation_id, role (user|assistant), content,
                               citations JSONB (num/pmid/title/url per cited article), created_at
```

### Embedding model

`BAAI/bge-small-en-v1.5` (384-dimensional vectors) via [fastembed](https://github.com/qdrant/fastembed). Used for both indexing article chunks and embedding queries at search/ask time.

### Claude models

| Task | Default model | Override env var |
|---|---|---|
| PubMed query generation (extended thinking) | `claude-opus-4-6` | `QUERY_BUILDER_MODEL` |
| Starter question generation | `claude-opus-4-6` | `STARTER_QUESTIONS_MODEL` |
| Collection chat | `claude-opus-4-6` | `DEFAULT_CHAT_MODEL` |

The chat model can also be switched at runtime from the collection page (Opus 4.6 / Sonnet 4.6 / Haiku 4.5); the preference is persisted in `localStorage`.

All tunable values — model IDs, system prompts, timeouts, chunk sizes, token limits — live in `config.py`.

### Logging

Logs go to the console and `logs/app.log` (rotating, 10 MB × 5 backups). Verbosity is controlled by the `LOG_LEVEL` env var.

---

## Project structure

```
app.py                   # Flask routes and all business logic
config.py                # Central configuration — models, prompts, timeouts, limits
db.py                    # PostgreSQL / pgvector layer
requirements.txt
Dockerfile
.dockerignore

templates/
  base.html              # Shared layout (progress bar, breadcrumbs, nav)
  index.html             # Search + relevance review panel
  collections.html       # Collection list
  collection.html        # Collection detail + 3-column layout (articles | conversations | chat)

static/
  base.css / base.js     # Shared styles and progress bar
  index.css / index.js   # Search page and review panel
  collections.css / collections.js
  collection.css / collection.js   # Chat UI, conversation sidebar, history loader

tests/
  test_app.py            # Unit tests — routes, pure functions, helpers (pytest)
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

MIT License. See [LICENSE](LICENSE) for the full text.