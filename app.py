import io
import json
import logging
import logging.handlers
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import numpy as np
import requests
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor
from flask import Flask, Response, render_template, request, stream_with_context
from dotenv import load_dotenv

import db
import config as cfg

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    root = logging.getLogger("pubmed")
    root.setLevel(level)
    if not root.handlers:          # avoid duplicate handlers on Flask reloader restart
        root.addHandler(file_handler)
        root.addHandler(console_handler)

    root.info("Logging initialised | level=%s | log=%s", level_name, log_dir / "app.log")
    return root

_log       = setup_logging()
_api_log   = logging.getLogger("pubmed.api")
_claude_log = logging.getLogger("pubmed.claude")
_embed_log = logging.getLogger("pubmed.embed")

_starter_questions_cache: dict[int, list[str]] = {}
_SSE_MIMETYPE = "text/event-stream"

app = Flask(__name__)
app.jinja_env.globals["static_version"] = cfg.STATIC_VERSION
client = anthropic.Anthropic()

# ── Embeddings ────────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(cfg.EMBEDDING_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    _embed_log.debug("Embedding %d text(s)", len(texts))
    t0 = time.perf_counter()
    result = [np.array(v, dtype=np.float32) for v in get_embedder().embed(texts)]
    elapsed = time.perf_counter() - t0
    dim = len(result[0]) if result else 0
    _embed_log.info("Embedded texts=%d dim=%d duration=%.3fs", len(texts), dim, elapsed)
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── PubMed helpers ────────────────────────────────────────────────────────────

def build_pubmed_query(english_query: str) -> str:
    _claude_log.debug("op=build_query query=%r", english_query[:120])
    t0 = time.perf_counter()
    with client.messages.stream(
        model=cfg.QUERY_BUILDER_MODEL,
        max_tokens=cfg.MAX_TOKENS_PUBMED_QUERY,
        thinking={"type": "adaptive"},
        system=cfg.PROMPT_PUBMED_QUERY,
        messages=[{"role": "user", "content": english_query}],
    ) as stream:
        msg = stream.get_final_message()
        text_block = next(b for b in msg.content if b.type == "text")
        result = text_block.text.strip()
    elapsed = time.perf_counter() - t0
    _claude_log.info(
        "op=build_query model=%s in_tokens=%d out_tokens=%d duration=%.2fs",
        cfg.QUERY_BUILDER_MODEL, msg.usage.input_tokens, msg.usage.output_tokens, elapsed,
    )
    _claude_log.debug("op=build_query result=%r", result)
    return result


# ── NCBI request helper with retry / back-off ─────────────────────────────────

def _ncbi_get(url: str, params: dict, timeout: int, max_retries: int = 4) -> requests.Response:
    """GET with exponential back-off on transient NCBI errors (429, 5xx, timeouts).

    Delays: 1 s, 2 s, 4 s, 8 s (doubles each attempt, capped at 30 s).
    Raises the last exception if all retries are exhausted.
    """
    if cfg.PUBMED_API_KEY:
        params = {**params, "api_key": cfg.PUBMED_API_KEY}
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(response=resp)
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            _ncbi_handle_retry(url, attempt, max_retries, delay, exc)
            time.sleep(delay)
            delay = min(delay * 2, cfg.NCBI_BACKOFF_MAX)


def _ncbi_handle_retry(url: str, attempt: int, max_retries: int, delay: float, exc: Exception):
    """Log and raise on final attempt, or log a warning and allow retry."""
    if attempt == max_retries:
        _api_log.error(
            "op=ncbi_get url=%s attempt=%d/%d FAILED: %s",
            url, attempt + 1, max_retries + 1, exc,
        )
        raise exc
    _api_log.warning(
        "op=ncbi_get url=%s attempt=%d/%d retrying in %.0fs: %s",
        url, attempt + 1, max_retries + 1, delay, exc,
    )


def search_pubmed(query: str, max_results: int = 25) -> list[str]:
    _api_log.debug("op=esearch query=%r max=%d", query[:120], max_results)
    t0 = time.perf_counter()
    resp = _ncbi_get(
        f"{cfg.PUBMED_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"},
        timeout=cfg.TIMEOUT_ESEARCH,
    )
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]
    elapsed = time.perf_counter() - t0
    _api_log.info("op=esearch status=%d results=%d duration=%.2fs", resp.status_code, len(ids), elapsed)
    return ids


def _parse_authors(article_node) -> str:
    authors = []
    for author in article_node.findall(".//AuthorList/Author"):
        last = author.findtext("LastName") or ""
        initials = author.findtext("Initials") or ""
        collective = author.findtext("CollectiveName") or ""
        name = f"{last} {initials}".strip() if last else collective
        if name:
            authors.append(name)
    author_str = ", ".join(authors[:cfg.AUTHORS_DISPLAY_MAX])
    if len(authors) > cfg.AUTHORS_DISPLAY_MAX:
        author_str += " et al."
    return author_str or "Unknown"


def _parse_abstract(article_node) -> str:
    parts = []
    for ab in article_node.findall(".//Abstract/AbstractText"):
        text = (ab.text or "").strip()
        if not text:
            continue
        label = ab.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def _parse_article(pmid: str, node) -> dict:
    article = node.find(".//MedlineCitation/Article")
    title = (article.findtext("ArticleTitle") or "No title").rstrip(".")
    journal = (
        article.findtext(".//Journal/Title")
        or article.findtext(".//Journal/ISOAbbreviation")
        or ""
    )
    year = (
        article.findtext(".//Journal/JournalIssue/PubDate/Year")
        or article.findtext(".//Journal/JournalIssue/PubDate/MedlineDate")
        or ""
    )[:4]
    return {
        "pmid": pmid,
        "title": title,
        "authors": _parse_authors(article),
        "journal": journal,
        "year": year,
        "abstract": _parse_abstract(article),
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    _api_log.debug("op=efetch pmids=%d", len(pmids))
    t0 = time.perf_counter()
    resp = _ncbi_get(
        f"{cfg.PUBMED_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        timeout=cfg.TIMEOUT_EFETCH,
    )
    resp.raise_for_status()
    _api_log.debug("op=efetch status=%d bytes=%d", resp.status_code, len(resp.content))
    root = ET.fromstring(resp.content)

    nodes_by_pmid = {
        node.findtext(".//MedlineCitation/PMID"): node
        for node in root.findall(".//PubmedArticle")
        if node.findtext(".//MedlineCitation/PMID")
    }

    articles = [
        _parse_article(pmid, nodes_by_pmid[pmid])
        for pmid in pmids
        if pmid in nodes_by_pmid
    ]
    elapsed = time.perf_counter() - t0
    _api_log.info("op=efetch parsed=%d duration=%.2fs", len(articles), elapsed)
    return articles


# ── PMC full-text helpers ─────────────────────────────────────────────────────

def get_pmcids(pmids: list[str]) -> dict[str, str]:
    """Return {pmid: pmcid} for articles that have PMC full text."""
    if not pmids:
        return {}
    _api_log.debug("op=elink pmids=%d", len(pmids))
    t0 = time.perf_counter()
    resp = _ncbi_get(
        f"{cfg.PUBMED_BASE}/elink.fcgi",
        params={"dbfrom": "pubmed", "db": "pmc", "linkname": "pubmed_pmc", "id": pmids, "retmode": "json"},
        timeout=cfg.TIMEOUT_EFETCH,
    )
    resp.raise_for_status()
    data = resp.json()
    result = {}
    for linkset in data.get("linksets", []):
        ids = linkset.get("ids", [])
        pmid = str(ids[0]) if ids else None
        if not pmid:
            continue
        for lsdb in linkset.get("linksetdbs", []):
            if lsdb.get("linkname") == "pubmed_pmc" and lsdb.get("links"):
                result[pmid] = str(lsdb["links"][0])
    elapsed = time.perf_counter() - t0
    _api_log.info("op=elink status=%d pmc_hits=%d/%d duration=%.2fs", resp.status_code, len(result), len(pmids), elapsed)
    return result


def fetch_pmc_full_text(pmcid: str) -> str:
    """Fetch PMC article XML and return extracted plain text from the body."""
    _api_log.debug("op=pmc_fulltext pmcid=%s", pmcid)
    t0 = time.perf_counter()
    resp = _ncbi_get(
        f"{cfg.PUBMED_BASE}/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
        timeout=cfg.TIMEOUT_PMC_FULLTEXT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    body = root.find(".//body")
    target = body if body is not None else root

    parts = []
    for elem in target.iter():
        if elem.tag in {"p", "title"}:
            text = " ".join("".join(elem.itertext()).split())
            if len(text) > 20:
                parts.append(text)

    full_text = "\n\n".join(parts)
    elapsed = time.perf_counter() - t0
    _api_log.info(
        "op=pmc_fulltext pmcid=%s status=%d chars=%d paragraphs=%d duration=%.2fs",
        pmcid, resp.status_code, len(full_text), len(parts), elapsed,
    )
    return full_text


# ── Routes ────────────────────────────────────────────────────────────────────

def _clamp_max_results(raw: str) -> int:
    value = int(raw) if raw else cfg.MAX_RESULTS_DEFAULT
    return max(cfg.MAX_RESULTS_MIN, min(cfg.MAX_RESULTS_MAX, (value // cfg.MAX_RESULTS_MIN) * cfg.MAX_RESULTS_MIN))


def _attach_similarities(user_query: str, results: list[dict]):
    query_emb = embed_texts([user_query])[0]
    article_texts = [f"{a['title']}. {a['abstract']}" for a in results]
    article_embs = embed_texts(article_texts)
    for art, emb in zip(results, article_embs):
        art["similarity"] = round(cosine_similarity(query_emb, emb), 3)


def _run_search(user_query: str, max_results: int) -> tuple[list[dict] | None, str | None, str | None]:
    """Return (results, pubmed_query, error)."""
    try:
        pubmed_query = build_pubmed_query(user_query)
        pmids = search_pubmed(pubmed_query, max_results)
        results = fetch_articles(pmids)
        if results:
            _attach_similarities(user_query, results)
        return results, pubmed_query, None
    except anthropic.AuthenticationError:
        return None, None, "Invalid Anthropic API key. Set ANTHROPIC_API_KEY in your .env file."
    except requests.RequestException as exc:
        return None, None, f"PubMed request failed: {exc}"
    except Exception as exc:
        return None, None, str(exc)


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    pubmed_query = None
    error = None
    user_query = ""
    max_results = cfg.MAX_RESULTS_DEFAULT

    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            max_results = _clamp_max_results(request.form.get("max_results", str(cfg.MAX_RESULTS_DEFAULT)))
            results, pubmed_query, error = _run_search(user_query, max_results)

    return render_template(
        "index.html",
        user_query=user_query,
        pubmed_query=pubmed_query,
        results=results,
        error=error,
        max_results=max_results,
    )


@app.route("/collections", methods=["GET"])
def collections():
    items = db.list_collections()
    return render_template("collections.html", collections=items)


@app.route("/collections", methods=["POST"])
def collections_save():
    """Receive selected articles, fetch PMC full text where available, chunk, embed, store."""
    payload = request.get_json(force=True)
    name = (payload.get("name") or "").strip()
    user_query = payload.get("user_query", "")
    pubmed_query = payload.get("pubmed_query", "")
    articles = payload.get("articles", [])

    if not name:
        return {"error": "Collection name is required."}, 400
    if not articles:
        return {"error": "No articles to save."}, 400

    try:
        pmids = [a["pmid"] for a in articles]
        pmcid_map = get_pmcids(pmids)
        cid = db.create_collection(name, user_query, pubmed_query)

        # Phase 1: fetch PMC full texts in parallel (I/O-bound network calls)
        max_workers = min(len(articles), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fetched = list(executor.map(_fetch_article_text, articles, [pmcid_map] * len(articles)))

        # Phase 2: persist article rows; collect chunks that still need embedding
        pending: list[tuple[str, list[str]]] = []  # [(pmid, chunks), ...]
        for art, full_text, pmcid in fetched:
            db.add_article(cid, art, has_full_text=bool(full_text), pmcid=pmcid)
            pmid = art["pmid"]
            if not db.chunks_exist(pmid):
                text_to_chunk = full_text or f"{art['title']}. {art['abstract']}"
                pending.append((pmid, chunk_text(text_to_chunk, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)))

        # Phase 3: embed all pending chunks in one batch, then save
        if pending:
            all_chunks = [c for _, chunks in pending for c in chunks]
            all_embeddings = embed_texts(all_chunks)
            offset = 0
            for pmid, chunks in pending:
                n = len(chunks)
                db.save_chunks(pmid, chunks, all_embeddings[offset:offset + n])
                offset += n

        return {"id": cid}
    except Exception as exc:
        return {"error": str(exc)}, 500


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _save_fetch_phase(articles: list[dict], pmcid_map: dict):
    """Generator — yields SSE fetch events; returns (fetched, counts) via StopIteration.value."""
    total = len(articles)
    fetched: list[tuple[dict, str | None, str | None]] = []
    counts = {"full_text": 0, "abstract": 0, "fallback": 0}

    with ThreadPoolExecutor(max_workers=min(total, 8)) as executor:
        futures = {executor.submit(_fetch_article_text, art, pmcid_map): art for art in articles}
        for future in as_completed(futures):
            art, full_text, pmcid = future.result()
            fetched.append((art, full_text, pmcid))
            had_pmcid = bool(pmcid_map.get(futures[future]["pmid"]))
            if full_text:
                counts["full_text"] += 1
            elif had_pmcid:
                counts["fallback"] += 1
            else:
                counts["abstract"] += 1
            yield _sse({"type": "fetch", "done": len(fetched), "total": total, **counts})

    return fetched, counts


def _save_persist_phase(cid: int, fetched: list) -> list[tuple[str, list[str]]]:
    """Insert article rows; return (pmid, chunks) pairs that still need embedding."""
    pending: list[tuple[str, list[str]]] = []
    for art, full_text, pmcid in fetched:
        db.add_article(cid, art, has_full_text=bool(full_text), pmcid=pmcid)
        pmid = art["pmid"]
        if not db.chunks_exist(pmid):
            text_to_chunk = full_text or f"{art['title']}. {art['abstract']}"
            pending.append((pmid, chunk_text(text_to_chunk, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)))
    return pending


def _save_embed_phase(pending: list[tuple[str, list[str]]], counts: dict):
    """Generator — yields SSE embedding event then embeds and saves all pending chunks."""
    if not pending:
        return
    yield _sse({"type": "embedding", "count": len(pending),
                "full_text": counts["full_text"],
                "abstract": counts["abstract"] + counts["fallback"]})
    all_chunks = [c for _, chunks in pending for c in chunks]
    all_embeddings = embed_texts(all_chunks)
    offset = 0
    for pmid, chunks in pending:
        n = len(chunks)
        db.save_chunks(pmid, chunks, all_embeddings[offset:offset + n])
        offset += n


@app.route("/collections/save-stream", methods=["POST"])
def collections_save_stream():
    """SSE endpoint: save collection with per-article fetch + embedding progress."""
    payload = request.get_json(force=True)
    name = (payload.get("name") or "").strip()
    user_query = payload.get("user_query", "")
    pubmed_query = payload.get("pubmed_query", "")
    articles = payload.get("articles", [])

    if not name:
        return {"error": "Collection name is required."}, 400
    if not articles:
        return {"error": "No articles to save."}, 400

    def generate():
        try:
            pmcid_map = get_pmcids([a["pmid"] for a in articles])
            cid = db.create_collection(name, user_query, pubmed_query)
            fetched, counts = yield from _save_fetch_phase(articles, pmcid_map)
            pending = _save_persist_phase(cid, fetched)
            yield from _save_embed_phase(pending, counts)
            yield _sse({"type": "done", "id": cid, "count": len(articles), **counts})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return Response(
        stream_with_context(generate()),
        mimetype=_SSE_MIMETYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _try_fetch_full_text(pmcid: str) -> tuple[str | None, str | None]:
    """Return (full_text, pmcid) — pmcid is set to None on fetch failure."""
    try:
        return fetch_pmc_full_text(pmcid), pmcid
    except Exception:
        return None, None


def _fetch_article_text(art: dict, pmcid_map: dict) -> tuple[dict, str | None, str | None]:
    """Resolve PMC full text for one article (called in parallel).

    Returns (art, full_text, resolved_pmcid).
    """
    pmcid = pmcid_map.get(art["pmid"])
    if pmcid:
        full_text, resolved_pmcid = _try_fetch_full_text(pmcid)
    else:
        full_text, resolved_pmcid = None, None
    return art, full_text, resolved_pmcid


def generate_starter_questions(user_query: str, articles: list[dict]) -> list[str]:
    """Generate 4 opening questions for a collection using its topic and article titles."""
    titles = "\n".join(f"- {a['title']}" for a in articles[:10])
    _claude_log.debug("op=starter_questions topic=%r articles=%d", user_query[:80], len(articles))
    t0 = time.perf_counter()
    try:
        msg = client.messages.create(
            model=cfg.STARTER_QUESTIONS_MODEL,
            max_tokens=cfg.MAX_TOKENS_STARTER_QS,
            system=cfg.PROMPT_STARTER_QUESTIONS,
            messages=[{"role": "user", "content": f"Topic: {user_query}\n\nArticles:\n{titles}"}],
        )
        questions = json.loads(msg.content[0].text.strip())[:4]
        elapsed = time.perf_counter() - t0
        _claude_log.info(
            "op=starter_questions in_tokens=%d out_tokens=%d questions=%d duration=%.2fs",
            msg.usage.input_tokens, msg.usage.output_tokens, len(questions), elapsed,
        )
        return questions
    except Exception as exc:
        _claude_log.warning("op=starter_questions failed: %s", exc)
        return []


@app.route("/collections/<int:cid>", methods=["GET"])
def collection_detail(cid: int):
    collection = db.get_collection(cid)
    if not collection:
        return "Collection not found", 404
    articles = db.get_collection_articles(cid)
    return render_template(
        "collection.html",
        collection=collection,
        articles=articles,
    )


@app.route("/collections/<int:cid>/starter-questions", methods=["GET"])
def collection_starter_questions(cid: int):
    if cid not in _starter_questions_cache:
        collection = db.get_collection(cid)
        if not collection:
            return {"questions": []}
        articles = db.get_collection_articles(cid)
        _starter_questions_cache[cid] = generate_starter_questions(
            collection["user_query"], articles
        )
    return {"questions": _starter_questions_cache[cid]}



def _sse_emit(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _parse_suggestions(raw: str) -> list[str]:
    try:
        return json.loads(raw.strip())[:4]
    except Exception:
        return []


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for one API call at Anthropic list prices."""
    pricing = cfg.MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def _stream_delimited_response(
    model: str, context: str, question: str, delim: str, emit
) -> tuple[list[str], str, dict]:
    """Stream a Claude response split by *delim*.

    Returns (sse_events, post_delimiter_text, usage_dict).
    usage_dict has keys: input_tokens, output_tokens.
    """
    pre = ""
    post = ""
    found = False
    events: list[str] = []

    with client.messages.stream(
        model=model,
        max_tokens=cfg.MAX_TOKENS_RAG_RESPONSE,
        system=cfg.PROMPT_RAG_SYSTEM,
        messages=[{"role": "user", "content": f"Articles:\n\n{context}\n\nQuestion: {question}"}],
    ) as stream:
        for text in stream.text_stream:
            pre, post, found, chunk_events = _advance_delimiter_state(
                pre, post, found, text, delim, emit
            )
            events.extend(chunk_events)
        try:
            final_msg = stream.get_final_message()
            usage = {
                "input_tokens":  final_msg.usage.input_tokens,
                "output_tokens": final_msg.usage.output_tokens,
            }
        except Exception as exc:
            _claude_log.warning("op=stream_usage failed: %s", exc)
            usage = {"input_tokens": 0, "output_tokens": 0}

    if not found and pre:
        events.append(emit({"text": pre}))

    return events, post, usage


def _advance_delimiter_state(
    pre: str, post: str, found: bool, text: str, delim: str, emit
) -> tuple[str, str, bool, list[str]]:
    """Process one streamed token; return updated (pre, post, found, new_events)."""
    events: list[str] = []
    if found:
        return pre, post + text, found, events

    pre += text
    if delim in pre:
        idx = pre.index(delim)
        safe = pre[:idx]
        post = pre[idx + len(delim):]
        pre = ""
        if safe:
            events.append(emit({"text": safe}))
        return pre, post, True, events

    safe_len = max(0, len(pre) - len(delim) + 1)
    if safe_len:
        events.append(emit({"text": pre[:safe_len]}))
        pre = pre[safe_len:]
    return pre, post, found, events


def _build_rag_context(top_chunks: list[dict]) -> tuple[str, list[dict]]:
    """Return (context_string, deduplicated_citations).

    Each unique article is assigned one citation number. Chunks from the same
    article share that number so Claude's inline [N] markers always match the
    numbers shown in the Sources section.
    """
    context_parts = []
    pmid_to_num: dict[str, int] = {}
    citations = []
    for chunk in top_chunks:
        pmid = chunk["pmid"]
        if pmid not in pmid_to_num:
            pmid_to_num[pmid] = len(citations) + 1
            citations.append(chunk)
        num = pmid_to_num[pmid]
        context_parts.append(
            f"[{num}] {chunk['title']}\n"
            f"Authors: {chunk['authors']}\n"
            f"Journal: {chunk['journal']} ({chunk['year']})\n"
            f"Excerpt: {chunk['chunk_text']}"
        )
    return "\n\n".join(context_parts), citations


def _extract_answer_from_events(events: list[str]) -> str:
    """Reconstruct plain answer text from SSE event strings."""
    parts = []
    for event in events:
        if not event.startswith("data: "):
            continue
        try:
            payload = json.loads(event[6:].rstrip("\n"))
            if "text" in payload:
                parts.append(payload["text"])
        except Exception:
            pass
    return "".join(parts)


@app.route("/collections/<int:cid>/conversations", methods=["GET"])
def collection_conversations(cid: int):
    return db.list_conversations(cid)


@app.route("/conversations/<int:vid>/messages", methods=["GET"])
def conversation_messages(vid: int):
    return db.get_conversation_messages(vid)


@app.route("/conversations/<int:vid>/rename", methods=["PATCH"])
def conversation_rename(vid: int):
    data  = request.get_json(force=True)
    title = (data.get("title") or "").strip()
    if not title:
        return {"error": "Title cannot be empty."}, 400
    db.rename_conversation(vid, title)
    return {"ok": True}


@app.route("/conversations/<int:vid>/delete", methods=["POST"])
def conversation_delete(vid: int):
    db.delete_conversation(vid)
    return {"ok": True}


@app.route("/collections/<int:cid>/ask", methods=["POST"])
def collection_ask(cid: int):
    """SSE endpoint: embeds question, retrieves top-k chunks, streams RAG answer."""
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return {"error": "No question provided."}, 400
    model = data.get("model", cfg.DEFAULT_CHAT_MODEL)
    if model not in cfg.ALLOWED_CHAT_MODELS:
        model = cfg.DEFAULT_CHAT_MODEL
    conversation_id = data.get("conversation_id")

    q_emb = embed_texts([question])[0]
    top_chunks = db.semantic_search(cid, q_emb, k=cfg.SEMANTIC_SEARCH_K)

    if not top_chunks:
        # Still persist the unanswered question so the conversation exists
        if conversation_id is None:
            conversation_id = db.create_conversation(cid, question)
        db.add_message(conversation_id, "user", question)
        db.add_message(conversation_id, "assistant", "No articles found in this collection.")

        def empty():
            yield f"data: {json.dumps({'text': 'No articles found in this collection.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'citations': [], 'conversation_id': conversation_id})}\n\n"
        return Response(empty(), mimetype=_SSE_MIMETYPE,
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Create / continue conversation
    if conversation_id is None:
        conversation_id = db.create_conversation(cid, question)
    db.add_message(conversation_id, "user", question)

    context, citations = _build_rag_context(top_chunks)

    # Slim, serialisable form — num field lets the frontend map [N] → URL
    stored_citations = [
        {
            "num": i + 1,
            "pmid": c["pmid"],
            "title": c["title"],
            "url": c["url"],
            "journal": c["journal"],
            "year": c["year"],
            "similarity": c["similarity"],
        }
        for i, c in enumerate(citations)
    ]

    _DELIM = cfg.RAG_DELIMITER

    def generate():
        t0 = time.perf_counter()
        _claude_log.debug(
            "op=collection_ask cid=%d chunks=%d question=%r",
            cid, len(top_chunks), question[:120],
        )
        try:
            answer_text, suggestions_json, usage = _stream_delimited_response(
                model, context, question, _DELIM, _sse_emit
            )
        except Exception as exc:
            _claude_log.error("op=collection_ask FAILED cid=%d: %s", cid, exc)
            yield _sse_emit({"error": str(exc)})
            return

        yield from answer_text

        full_answer = _extract_answer_from_events(answer_text)
        db.add_message(conversation_id, "assistant", full_answer, stored_citations)

        suggestions = _parse_suggestions(suggestions_json)
        elapsed = time.perf_counter() - t0
        cost = _compute_cost(model, usage["input_tokens"], usage["output_tokens"])
        usage_info = {
            "model":         model,
            "input_tokens":  usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd":      round(cost, 6),
        }
        _claude_log.info(
            "op=collection_ask cid=%d vid=%d citations=%d suggestions=%d "
            "in_tok=%d out_tok=%d cost_usd=%.6f duration=%.2fs",
            cid, conversation_id, len(stored_citations), len(suggestions),
            usage["input_tokens"], usage["output_tokens"], cost, elapsed,
        )
        yield f"data: {json.dumps({'done': True, 'citations': stored_citations, 'suggestions': suggestions, 'conversation_id': conversation_id, 'usage': usage_info})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype=_SSE_MIMETYPE,
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/collections/<int:cid>/delete", methods=["POST"])
def collection_delete(cid: int):
    db.delete_collection(cid)
    return {"ok": True}


# ── Conversation export ───────────────────────────────────────────────────────

def _rtf_esc(text: str) -> str:
    """Escape a string for inclusion in an RTF document body."""
    out = []
    for ch in text:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '{':
            out.append('\\{')
        elif ch == '}':
            out.append('\\}')
        elif ord(ch) > 127:
            out.append(f'\\u{ord(ch)}?')
        else:
            out.append(ch)
    return ''.join(out)


def _rtf_hyperlink(url: str, text: str) -> str:
    """Return an RTF field that renders *text* as a clickable hyperlink to *url*."""
    safe_url = url.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
    return (
        r'{\field{\*\fldinst{HYPERLINK "' + safe_url + r'"}}' +
        r'{\fldrslt \cf2\ul ' + _rtf_esc(text) + r'}}'
    )


def _rtf_line_with_links(text: str, url_map: dict) -> str:
    """Escape *text* for RTF, converting [N] markers to hyperlinks via *url_map*."""
    parts = re.split(r'(\[\d+\])', text)
    out = []
    for part in parts:
        m = re.match(r'^\[(\d+)\]$', part)
        url = m and url_map.get(int(m.group(1)))
        out.append(_rtf_hyperlink(url, part) if url else _rtf_esc(part))
    return ''.join(out)


def _rtf_citation_line(c: dict) -> str:
    """Return the RTF line string for a single citation entry."""
    num     = c.get('num', '')
    url     = c.get('url', '')
    ctitle  = c.get('title', '')
    journal = c.get('journal')
    meta    = ''
    if journal:
        year = c.get('year')
        meta = f" - {journal} ({year})" if year else f" - {journal}"
    link = _rtf_hyperlink(url, ctitle) if url else _rtf_esc(ctitle)
    return r'\pard [' + _rtf_esc(str(num)) + '] ' + link + _rtf_esc(meta) + r'\par'


def _rtf_user_turn(msg: dict) -> str:
    """Return the RTF line string for a user message."""
    return r'\pard\cf1\b You:\b0\cf0 ' + _rtf_esc(msg['content']) + r'\par'


def _rtf_assistant_turn(msg: dict) -> list[str]:
    """Return the RTF line strings for an assistant message with optional citations."""
    citations = msg.get('citations') or []
    url_map   = {c['num']: c['url'] for c in citations if 'num' in c and 'url' in c}
    result    = [r'\pard\b Claude:\b0\par']
    for line in msg['content'].split('\n'):
        result.append(r'\pard ' + _rtf_line_with_links(line, url_map) + r'\par')
    if citations:
        result.append(r'\pard\par\pard\b\fs18 Sources\b0\fs20\par')
        for c in citations:
            result.append(_rtf_citation_line(c))
    return result


def _build_rtf(title: str, collection_name: str, created_at: str, messages: list[dict]) -> str:
    lines = [
        r'{\rtf1\ansi\ansicpg1252\deff0',
        r'{\fonttbl{\f0\fswiss\fcharset0 Arial;}}',
        r'{\colortbl;\red15\green52\blue96;\red5\green99\blue193;}',  # \cf1=navy \cf2=link-blue
        r'\f0\fs20',
        r'\pard\b\fs28 ' + _rtf_esc(title) + r'\b0\fs20\par',
        r'\pard\i Collection: ' + _rtf_esc(collection_name) + r'\i0\par',
        r'\pard\i Date: ' + _rtf_esc(created_at[:10]) + r'\i0\par',
        r'\pard\par',
    ]
    for msg in messages:
        if msg['role'] == 'user':
            lines.append(_rtf_user_turn(msg))
        else:
            lines.extend(_rtf_assistant_turn(msg))
        lines.append(r'\pard\par')
    lines.append('}')
    return '\n'.join(lines)


# ── Docx hyperlink helpers ────────────────────────────────────────────────────

def _docx_add_hyperlink(para, url: str, text: str):
    """Append a clickable hyperlink run to *para*."""
    r_id = para.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    rpr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rpr.append(u)
    run.append(rpr)

    t = OxmlElement("w:t")
    t.text = text
    run.append(t)
    hl.append(run)
    para._p.append(hl)


def _docx_append_with_links(para, text: str, url_map: dict):
    """Append *text* to *para*, converting [N] markers to hyperlinks."""
    for part in re.split(r'(\[\d+\])', text):
        m = re.match(r'^\[(\d+)\]$', part)
        url = m and url_map.get(int(m.group(1)))
        if url:
            _docx_add_hyperlink(para, url, part)
        else:
            para.add_run(part)


def _render_citation(doc, c: dict):
    """Append a single citation paragraph to *doc*."""
    item = doc.add_paragraph()
    item.add_run(f'[{c.get("num", "")}] ')
    url    = c.get('url', '')
    ctitle = c.get('title', '')
    if url:
        _docx_add_hyperlink(item, url, ctitle)
    else:
        item.add_run(ctitle)
    journal = c.get('journal')
    if journal:
        year    = c.get('year')
        suffix  = f" \u2013 {journal} ({year})" if year else f" \u2013 {journal}"
        item.add_run(suffix).italic = True


def _render_user_turn(doc, msg: dict):
    """Append a user message paragraph to *doc*."""
    p     = doc.add_paragraph()
    label = p.add_run('You:  ')
    label.bold = True
    label.font.color.rgb = RGBColor(0x0F, 0x34, 0x60)
    p.add_run(msg['content'])


def _render_assistant_turn(doc, msg: dict):
    """Append an assistant message (with optional citations) to *doc*."""
    citations = msg.get('citations') or []
    url_map   = {c['num']: c['url'] for c in citations if 'num' in c and 'url' in c}

    lines = msg['content'].split('\n')
    p = doc.add_paragraph()
    p.add_run('Claude:  ').bold = True
    _docx_append_with_links(p, lines[0], url_map)
    for line in lines[1:]:
        p = doc.add_paragraph()
        _docx_append_with_links(p, line, url_map)

    if citations:
        doc.add_paragraph().add_run('Sources:').bold = True
        for c in citations:
            _render_citation(doc, c)


def _build_docx(title: str, collection_name: str, created_at: str, messages: list[dict]) -> io.BytesIO:
    doc = Document()
    doc.add_heading(title, level=1)

    meta = doc.add_paragraph()
    meta.add_run(f'Collection: {collection_name}  |  {created_at[:10]}').italic = True
    doc.add_paragraph()

    for msg in messages:
        if msg['role'] == 'user':
            _render_user_turn(doc, msg)
        else:
            _render_assistant_turn(doc, msg)
        doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf


@app.route("/conversations/<int:vid>/export", methods=["GET"])
def conversation_export(vid: int):
    fmt = request.args.get("format", "docx").lower()
    if fmt not in ("docx", "rtf"):
        return {"error": "format must be 'docx' or 'rtf'"}, 400

    conv = db.get_conversation(vid)
    if not conv:
        return "Conversation not found", 404

    messages = db.get_conversation_messages(vid)
    if not messages:
        return "No messages to export", 404

    safe_name = (
        "".join(c for c in conv['title'] if c.isalnum() or c in ' -_')[:60].strip()
        or "conversation"
    )

    if fmt == "docx":
        buf = _build_docx(conv['title'], conv['collection_name'], conv['created_at'], messages)
        return Response(
            buf.read(),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.docx"'},
        )

    rtf_text = _build_rtf(conv['title'], conv['collection_name'], conv['created_at'], messages)
    return Response(
        rtf_text.encode("ascii", errors="replace"),
        mimetype="application/rtf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.rtf"'},
    )


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db.init_db()
    app.run(host="0.0.0.0", debug=True, port=8080)