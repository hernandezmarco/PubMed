import json
import logging
import logging.handlers
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import anthropic
import numpy as np
import requests
from flask import Flask, Response, render_template, request, stream_with_context
from dotenv import load_dotenv

import db

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

app = Flask(__name__)
client = anthropic.Anthropic()

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ── Embeddings ────────────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
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
        model="claude-opus-4-6",
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=(
            "You are a biomedical librarian expert in PubMed search syntax. "
            "Convert the user's plain-English research question into an optimal PubMed "
            "query using MeSH terms, field tags ([MeSH Terms], [Title/Abstract], etc.), "
            "and Boolean operators (AND, OR, NOT). "
            "Return ONLY the raw query string — no explanation, no markdown, no quotes."
        ),
        messages=[{"role": "user", "content": english_query}],
    ) as stream:
        msg = stream.get_final_message()
        text_block = next(b for b in msg.content if b.type == "text")
        result = text_block.text.strip()
    elapsed = time.perf_counter() - t0
    _claude_log.info(
        "op=build_query model=claude-opus-4-6 in_tokens=%d out_tokens=%d duration=%.2fs",
        msg.usage.input_tokens, msg.usage.output_tokens, elapsed,
    )
    _claude_log.debug("op=build_query result=%r", result)
    return result


def search_pubmed(query: str, max_results: int = 25) -> list[str]:
    _api_log.debug("op=esearch query=%r max=%d", query[:120], max_results)
    t0 = time.perf_counter()
    resp = requests.get(
        f"{PUBMED_BASE}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "relevance"},
        timeout=10,
    )
    resp.raise_for_status()
    ids = resp.json()["esearchresult"]["idlist"]
    elapsed = time.perf_counter() - t0
    _api_log.info("op=esearch status=%d results=%d duration=%.2fs", resp.status_code, len(ids), elapsed)
    return ids


def fetch_articles(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    _api_log.debug("op=efetch pmids=%d", len(pmids))
    t0 = time.perf_counter()
    resp = requests.get(
        f"{PUBMED_BASE}/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        timeout=15,
    )
    resp.raise_for_status()
    _api_log.debug("op=efetch status=%d bytes=%d", resp.status_code, len(resp.content))
    root = ET.fromstring(resp.content)

    nodes_by_pmid = {}
    for node in root.findall(".//PubmedArticle"):
        pmid = node.findtext(".//MedlineCitation/PMID")
        if pmid:
            nodes_by_pmid[pmid] = node

    articles = []
    for pmid in pmids:
        node = nodes_by_pmid.get(pmid)
        if node is None:
            continue

        article = node.find(".//MedlineCitation/Article")
        title = (article.findtext("ArticleTitle") or "No title").rstrip(".")

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName") or ""
            initials = author.findtext("Initials") or ""
            collective = author.findtext("CollectiveName") or ""
            name = f"{last} {initials}".strip() if last else collective
            if name:
                authors.append(name)
        author_str = ", ".join(authors[:6])
        if len(authors) > 6:
            author_str += " et al."

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

        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = (ab.text or "").strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n\n".join(abstract_parts) if abstract_parts else ""

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "authors": author_str or "Unknown",
                "journal": journal,
                "year": year,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
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
    resp = requests.get(
        f"{PUBMED_BASE}/elink.fcgi",
        params={"dbfrom": "pubmed", "db": "pmc", "linkname": "pubmed_pmc", "id": pmids, "retmode": "json"},
        timeout=15,
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
    resp = requests.get(
        f"{PUBMED_BASE}/efetch.fcgi",
        params={"db": "pmc", "id": pmcid, "rettype": "full", "retmode": "xml"},
        timeout=30,
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

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    pubmed_query = None
    error = None
    user_query = ""
    max_results = 25

    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            try:
                max_results = int(request.form.get("max_results", 25))
                max_results = max(25, min(200, (max_results // 25) * 25))
                pubmed_query = build_pubmed_query(user_query)
                pmids = search_pubmed(pubmed_query, max_results)
                results = fetch_articles(pmids)

                # Compute cosine similarity: user query vs each article
                if results:
                    query_emb = embed_texts([user_query])[0]
                    article_texts = [f"{a['title']}. {a['abstract']}" for a in results]
                    article_embs = embed_texts(article_texts)
                    for art, emb in zip(results, article_embs):
                        art["similarity"] = round(cosine_similarity(query_emb, emb), 3)

            except anthropic.AuthenticationError:
                error = "Invalid Anthropic API key. Set ANTHROPIC_API_KEY in your .env file."
            except requests.RequestException as exc:
                error = f"PubMed request failed: {exc}"
            except Exception as exc:
                error = str(exc)

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

        for art in articles:
            pmid = art["pmid"]
            pmcid = pmcid_map.get(pmid)

            # Attempt full-text download
            full_text = None
            if pmcid:
                try:
                    full_text = fetch_pmc_full_text(pmcid)
                except Exception:
                    pmcid = None

            has_full_text = bool(full_text)
            db.add_article(cid, art, has_full_text=has_full_text, pmcid=pmcid)

            # Chunk + embed only if not already stored (dedup by PMID)
            if not db.chunks_exist(pmid):
                text_to_chunk = full_text if full_text else f"{art['title']}. {art['abstract']}"
                chunks = chunk_text(text_to_chunk)
                embeddings = embed_texts(chunks)
                db.save_chunks(pmid, chunks, embeddings)

        return {"id": cid}
    except Exception as exc:
        return {"error": str(exc)}, 500


def generate_starter_questions(user_query: str, articles: list[dict]) -> list[str]:
    """Generate 4 opening questions for a collection using its topic and article titles."""
    titles = "\n".join(f"- {a['title']}" for a in articles[:10])
    _claude_log.debug("op=starter_questions topic=%r articles=%d", user_query[:80], len(articles))
    t0 = time.perf_counter()
    try:
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=256,
            system=(
                "Generate exactly 4 concise research questions a user might want to ask about "
                "this collection. Return ONLY a JSON array of 4 strings, no other text."
            ),
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


@app.route("/collections/<int:cid>")
def collection_detail(cid: int):
    collection = db.get_collection(cid)
    if not collection:
        return "Collection not found", 404
    articles = db.get_collection_articles(cid)
    starter_questions = generate_starter_questions(collection["user_query"], articles)
    return render_template(
        "collection.html",
        collection=collection,
        articles=articles,
        starter_questions=starter_questions,
    )


@app.route("/collections/<int:cid>/ask", methods=["POST"])
def collection_ask(cid: int):
    """SSE endpoint: embeds question, retrieves top-k chunks, streams RAG answer."""
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return {"error": "No question provided."}, 400

    q_emb = embed_texts([question])[0]
    top_chunks = db.semantic_search(cid, q_emb, k=5)

    if not top_chunks:
        def empty():
            yield f"data: {json.dumps({'text': 'No articles found in this collection.'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'citations': []})}\n\n"
        return Response(empty(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Build context from chunks; deduplicate citations by PMID
    context_parts = []
    seen_pmids: set[str] = set()
    citations = []
    for i, chunk in enumerate(top_chunks, 1):
        context_parts.append(
            f"[{i}] {chunk['title']}\n"
            f"Authors: {chunk['authors']}\n"
            f"Journal: {chunk['journal']} ({chunk['year']})\n"
            f"Excerpt: {chunk['chunk_text']}"
        )
        if chunk["pmid"] not in seen_pmids:
            seen_pmids.add(chunk["pmid"])
            citations.append(chunk)

    context = "\n\n".join(context_parts)

    DELIM = "\n===\n"

    def generate():
        pre = ""        # text before the delimiter
        post = ""       # text after the delimiter (suggestions JSON)
        found = False
        t0 = time.perf_counter()
        _claude_log.debug(
            "op=collection_ask cid=%d chunks=%d question=%r",
            cid, len(top_chunks), question[:120],
        )

        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=2560,
            system=(
                "You are a biomedical research assistant. Answer the user's question using "
                "ONLY the numbered article excerpts provided. Cite sources inline as [1], [2], etc. "
                "Be concise and precise. If the excerpts lack sufficient information, say so.\n\n"
                "After your answer output a line containing only === followed immediately by a "
                "JSON array of exactly 4 concise follow-up questions the user might ask next, "
                "based on your answer. Example:\n"
                "===\n"
                "[\"Question one?\", \"Question two?\", \"Question three?\", \"Question four?\"]"
            ),
            messages=[{
                "role": "user",
                "content": f"Articles:\n\n{context}\n\nQuestion: {question}",
            }],
        ) as stream:
            for text in stream.text_stream:
                if found:
                    post += text
                    continue

                pre += text

                if DELIM in pre:
                    found = True
                    idx = pre.index(DELIM)
                    safe = pre[:idx]
                    post = pre[idx + len(DELIM):]
                    pre = ""
                    if safe:
                        yield f"data: {json.dumps({'text': safe})}\n\n"
                else:
                    # Emit safe portion; hold back enough chars for a partial delimiter match
                    safe_len = max(0, len(pre) - len(DELIM) + 1)
                    if safe_len:
                        yield f"data: {json.dumps({'text': pre[:safe_len]})}\n\n"
                        pre = pre[safe_len:]

        # Flush any remaining pre-delimiter text (delimiter never appeared)
        if not found and pre:
            yield f"data: {json.dumps({'text': pre})}\n\n"

        suggestions = []
        try:
            suggestions = json.loads(post.strip())[:4]
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        _claude_log.info(
            "op=collection_ask cid=%d citations=%d suggestions=%d duration=%.2fs",
            cid, len(citations), len(suggestions), elapsed,
        )

        yield f"data: {json.dumps({'done': True, 'citations': citations, 'suggestions': suggestions})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/collections/<int:cid>/delete", methods=["POST"])
def collection_delete(cid: int):
    db.delete_collection(cid)
    return {"ok": True}


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, port=8080)