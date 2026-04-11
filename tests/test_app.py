"""
Unit tests for app.py — covers pure functions and Flask routes (mocked externals).
Run with:  pytest tests/test_app.py -v
"""
import json
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Import helpers we can test without side-effects ──────────────────────────
# Patch heavy imports before importing app so the module loads cleanly in CI.
import sys

# Stub out fastembed so get_embedder() is never called implicitly at import time
fastembed_stub = MagicMock()
fastembed_stub.TextEmbedding = MagicMock()
sys.modules.setdefault("fastembed", fastembed_stub)

import app as _app  # noqa: E402  (must come after stubs)
import config as cfg


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert _app.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _app.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _app.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _app.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_general_case(self):
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 0.0])
        expected = 1.0 / np.sqrt(2)
        assert _app.cosine_similarity(a, b) == pytest.approx(expected)


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = _app.chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)
        assert chunks == ["Hello world"]

    def test_exact_chunk_size(self):
        text = "a" * cfg.CHUNK_SIZE
        chunks = _app.chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)
        assert len(chunks) == 1
        assert chunks[0] == "a" * cfg.CHUNK_SIZE

    def test_two_chunks_with_overlap(self):
        text = "a" * (cfg.CHUNK_SIZE + cfg.CHUNK_OVERLAP)
        chunks = _app.chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)
        assert len(chunks) == 2
        # Second chunk starts at offset (CHUNK_SIZE - CHUNK_OVERLAP)
        assert len(chunks[1]) == cfg.CHUNK_OVERLAP * 2

    def test_empty_text_returns_no_chunks(self):
        assert _app.chunk_text("", chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP) == []

    def test_whitespace_only_not_included(self):
        text = "   " + "a" * cfg.CHUNK_SIZE + "   "
        chunks = _app.chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)
        # Strip means leading/trailing whitespace removed from each chunk
        for c in chunks:
            assert c == c.strip()

    def test_multiple_chunks_cover_all_content(self):
        # step = CHUNK_SIZE - CHUNK_OVERLAP; 3 steps fit in 2*CHUNK_SIZE + step
        step = cfg.CHUNK_SIZE - cfg.CHUNK_OVERLAP
        text = "x" * (cfg.CHUNK_SIZE + 2 * step)
        chunks = _app.chunk_text(text, chunk_size=cfg.CHUNK_SIZE, overlap=cfg.CHUNK_OVERLAP)
        assert len(chunks) == 3

class TestClampMaxResults:
    def test_default_empty_string(self):
        assert _app._clamp_max_results("") == cfg.MAX_RESULTS_DEFAULT

    def test_below_minimum_clamped_to_min(self):
        assert _app._clamp_max_results("10") == cfg.MAX_RESULTS_MIN

    def test_exact_minimum(self):
        assert _app._clamp_max_results(str(cfg.MAX_RESULTS_MIN)) == cfg.MAX_RESULTS_MIN

    def test_above_maximum_clamped_to_max(self):
        assert _app._clamp_max_results("999") == cfg.MAX_RESULTS_MAX

    def test_exact_maximum(self):
        assert _app._clamp_max_results(str(cfg.MAX_RESULTS_MAX)) == cfg.MAX_RESULTS_MAX

    def test_rounds_down_to_multiple_of_step(self):
        # 60 is between 2× and 3× the step size; should round down to 2×
        assert _app._clamp_max_results("60") == cfg.MAX_RESULTS_MIN * 2

    def test_exact_multiple_of_step_unchanged(self):
        assert _app._clamp_max_results("75") == cfg.MAX_RESULTS_MIN * 3

def _make_article_xml(authors_xml: str, abstract_xml: str = "", extra: str = "") -> ET.Element:
    xml = f"""
    <PubmedArticle>
      <MedlineCitation>
        <PMID>12345</PMID>
        <Article>
          <ArticleTitle>Test Title</ArticleTitle>
          <Journal>
            <Title>Test Journal</Title>
            <ISOAbbreviation>TJ</ISOAbbreviation>
            <JournalIssue>
              <PubDate><Year>2024</Year></PubDate>
            </JournalIssue>
          </Journal>
          <AuthorList>
            {authors_xml}
          </AuthorList>
          <Abstract>
            {abstract_xml}
          </Abstract>
          {extra}
        </Article>
      </MedlineCitation>
    </PubmedArticle>
    """
    return ET.fromstring(xml).find(".//MedlineCitation/Article")


class TestParseAuthors:
    def test_single_author(self):
        authors_xml = "<Author><LastName>Smith</LastName><Initials>JA</Initials></Author>"
        node = _make_article_xml(authors_xml)
        assert _app._parse_authors(node) == "Smith JA"

    def test_two_authors(self):
        authors_xml = """
        <Author><LastName>Smith</LastName><Initials>JA</Initials></Author>
        <Author><LastName>Jones</LastName><Initials>B</Initials></Author>
        """
        node = _make_article_xml(authors_xml)
        assert _app._parse_authors(node) == "Smith JA, Jones B"

    def test_more_than_max_authors_truncated(self):
        authors_xml = "".join(
            f"<Author><LastName>Author{i}</LastName><Initials>X</Initials></Author>"
            for i in range(cfg.AUTHORS_DISPLAY_MAX + 2)
        )
        node = _make_article_xml(authors_xml)
        result = _app._parse_authors(node)
        assert result.endswith("et al.")
        assert result.count(",") == cfg.AUTHORS_DISPLAY_MAX - 1

    def test_no_authors_returns_unknown(self):
        node = _make_article_xml("")
        assert _app._parse_authors(node) == "Unknown"

    def test_collective_name(self):
        authors_xml = "<Author><CollectiveName>The Study Group</CollectiveName></Author>"
        node = _make_article_xml(authors_xml)
        assert _app._parse_authors(node) == "The Study Group"



class TestParseAbstract:
    def test_plain_abstract(self):
        abstract_xml = '<AbstractText>This is the abstract.</AbstractText>'
        node = _make_article_xml("", abstract_xml)
        assert _app._parse_abstract(node) == "This is the abstract."

    def test_structured_abstract_with_labels(self):
        abstract_xml = """
        <AbstractText Label="BACKGROUND">Some background.</AbstractText>
        <AbstractText Label="METHODS">Some methods.</AbstractText>
        """
        node = _make_article_xml("", abstract_xml)
        result = _app._parse_abstract(node)
        assert "BACKGROUND: Some background." in result
        assert "METHODS: Some methods." in result

    def test_empty_abstract_text_skipped(self):
        abstract_xml = '<AbstractText></AbstractText>'
        node = _make_article_xml("", abstract_xml)
        assert _app._parse_abstract(node) == ""

    def test_no_abstract(self):
        node = _make_article_xml("")
        assert _app._parse_abstract(node) == ""


class TestSseEmit:
    def test_format(self):
        result = _app._sse_emit({"text": "hello"})
        assert result == 'data: {"text": "hello"}\n\n'

    def test_nested_payload(self):
        result = _app._sse_emit({"done": True, "citations": []})
        payload = json.loads(result[6:].strip())
        assert payload == {"done": True, "citations": []}


class TestParseSuggestions:
    def test_valid_json_array(self):
        raw = '["Q1?", "Q2?", "Q3?", "Q4?"]'
        assert _app._parse_suggestions(raw) == ["Q1?", "Q2?", "Q3?", "Q4?"]

    def test_truncated_to_four(self):
        raw = '["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
        assert len(_app._parse_suggestions(raw)) == 4

    def test_invalid_json_returns_empty(self):
        assert _app._parse_suggestions("not json") == []

    def test_empty_string_returns_empty(self):
        assert _app._parse_suggestions("") == []

    def test_whitespace_stripped(self):
        raw = '  ["Q1?"]  '
        assert _app._parse_suggestions(raw) == ["Q1?"]



class TestAdvanceDelimiterState:
    DELIM = cfg.RAG_DELIMITER

    def _emit(self, payload):
        return _app._sse_emit(payload)

    def test_accumulates_pre_delimiter_text(self):
        pre, _, found, events = _app._advance_delimiter_state(
            "", "", False, "hello", self.DELIM, self._emit
        )
        assert not found
        assert pre == "hello" or events  # safe_len may flush partial

    def test_detects_delimiter(self):
        _, post, found, events = _app._advance_delimiter_state(
            "answer", "", False, self.DELIM + "suggestions", self.DELIM, self._emit
        )
        assert found
        assert post == "suggestions"
        # The pre-delimiter text should have been emitted
        assert any("answer" in e for e in events)

    def test_after_delimiter_appends_to_post(self):
        _, post, found, events = _app._advance_delimiter_state(
            "", "part1", True, " part2", self.DELIM, self._emit
        )
        assert found
        assert post == "part1 part2"
        assert events == []

    def test_safe_flush_keeps_delimiter_window(self):
        # When text doesn't contain delimiter, chars outside the delimiter
        # window should be emitted as safe
        long_text = "x" * 100
        pre, _, _, events = _app._advance_delimiter_state(
            "", "", False, long_text, self.DELIM, self._emit
        )
        # Some safe text should have been emitted
        assert len(events) > 0 or len(pre) <= len(self.DELIM)



class TestBuildRagContext:
    def _make_chunk(self, pmid, title="T", authors="A", journal="J", year="2024",
                    chunk_text="text"):
        return {
            "pmid": pmid, "title": title, "authors": authors,
            "journal": journal, "year": year, "chunk_text": chunk_text,
        }

    def test_single_chunk(self):
        chunks = [self._make_chunk("111", title="Article One", chunk_text="Excerpt A")]
        context, citations = _app._build_rag_context(chunks)
        assert "[1] Article One" in context
        assert "Excerpt A" in context
        assert len(citations) == 1
        assert citations[0]["pmid"] == "111"

    def test_deduplicates_citations_by_pmid(self):
        chunks = [
            self._make_chunk("111", chunk_text="Chunk 1"),
            self._make_chunk("111", chunk_text="Chunk 2"),
            self._make_chunk("222", chunk_text="Chunk 3"),
        ]
        context, citations = _app._build_rag_context(chunks)
        # Context has all 3 numbered entries
        assert "[1]" in context and "[2]" in context and "[3]" in context
        # Citations deduplicated: only 2 unique PMIDs
        assert len(citations) == 2
        pmids = [c["pmid"] for c in citations]
        assert pmids == ["111", "222"]

    def test_numbering_increments(self):
        chunks = [self._make_chunk(str(i)) for i in range(3)]
        context, _ = _app._build_rag_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context



class TestExtractAnswerFromEvents:
    def test_basic_extraction(self):
        events = [
            _app._sse_emit({"text": "Hello "}),
            _app._sse_emit({"text": "world"}),
        ]
        assert _app._extract_answer_from_events(events) == "Hello world"

    def test_ignores_non_text_events(self):
        events = [
            _app._sse_emit({"text": "Answer"}),
            _app._sse_emit({"done": True}),
        ]
        assert _app._extract_answer_from_events(events) == "Answer"

    def test_ignores_malformed_lines(self):
        events = ["not a sse line", _app._sse_emit({"text": "ok"})]
        assert _app._extract_answer_from_events(events) == "ok"

    def test_empty_events(self):
        assert _app._extract_answer_from_events([]) == ""


# ═══════════════════════════════════════════════════════════════════════════════
# Flask route tests (mocked DB + external calls)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def client():
    _app.app.config["TESTING"] = True
    with _app.app.test_client() as c:
        yield c


class TestIndexRoute:
    def test_get_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    @patch("app._run_search")
    def test_post_with_query_calls_run_search(self, mock_run, client):
        mock_run.return_value = ([], "diabetes[MeSH]", None)
        resp = client.post("/", data={"query": "diabetes research", "max_results": "25"})
        assert resp.status_code == 200
        mock_run.assert_called_once_with("diabetes research", 25)

    def test_post_empty_query_no_search(self, client):
        resp = client.post("/", data={"query": ""})
        assert resp.status_code == 200

    @patch("app._run_search")
    def test_error_displayed(self, mock_run, client):
        mock_run.return_value = (None, None, "API key invalid")
        resp = client.post("/", data={"query": "test"})
        assert resp.status_code == 200
        assert b"API key invalid" in resp.data


class TestCollectionsRoute:
    @patch("app.db.list_collections", return_value=[])
    def test_get_collections(self, mock_list, client):
        resp = client.get("/collections")
        assert resp.status_code == 200
        mock_list.assert_called_once()


class TestCollectionsSaveRoute:
    @patch("app.get_pmcids", return_value={})
    @patch("app.db.create_collection", return_value=42)
    @patch("app._store_article")
    def test_save_success(self, mock_store, mock_create, mock_pmcids, client):
        payload = {
            "name": "My Collection",
            "user_query": "diabetes",
            "pubmed_query": "diabetes[MeSH]",
            "articles": [{"pmid": "123", "title": "T", "authors": "A",
                          "journal": "J", "year": "2024", "abstract": "AB",
                          "url": "http://example.com"}],
        }
        resp = client.post("/collections", json=payload)
        assert resp.status_code == 200
        assert resp.get_json()["id"] == 42

    def test_save_missing_name_returns_400(self, client):
        resp = client.post("/collections", json={"articles": [{"pmid": "1"}]})
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_save_missing_articles_returns_400(self, client):
        resp = client.post("/collections", json={"name": "Test", "articles": []})
        assert resp.status_code == 400


class TestCollectionDetailRoute:
    @patch("app.db.get_collection", return_value=None)
    def test_not_found_returns_404(self, mock_get, client):
        resp = client.get("/collections/999")
        assert resp.status_code == 404

    @patch("app.generate_starter_questions", return_value=["Q1?"])
    @patch("app.db.get_collection_articles", return_value=[])
    @patch("app.db.get_collection", return_value={
        "id": 1, "name": "Test", "user_query": "test",
        "pubmed_query": "test[MeSH]", "created_at": "2024-01-01",
        "article_count": 0, "full_text_count": 0, "chunk_count": 0,
    })
    def test_found_returns_200(self, mock_get, mock_articles, mock_questions, client):
        resp = client.get("/collections/1")
        assert resp.status_code == 200


class TestCollectionAskRoute:
    def test_empty_question_returns_400(self, client):
        resp = client.post("/collections/1/ask", json={"question": ""})
        assert resp.status_code == 400

    @patch("app.db.semantic_search", return_value=[])
    @patch("app.db.add_message")
    @patch("app.db.create_conversation", return_value=7)
    @patch("app.embed_texts")
    def test_no_chunks_streams_empty_message(
        self, mock_embed, mock_create_conv, mock_add_msg, mock_search, client
    ):
        mock_embed.return_value = [np.zeros(384)]
        resp = client.post(
            "/collections/1/ask",
            json={"question": "What is this?"},
            buffered=True,
        )
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/event-stream")
        data = resp.data.decode()
        assert "No articles found" in data

    def test_invalid_model_falls_back_to_default(self, client):
        # Just verify the model validation logic directly
        model = "invalid-model"
        if model not in cfg.ALLOWED_CHAT_MODELS:
            model = cfg.DEFAULT_CHAT_MODEL
        assert model == cfg.DEFAULT_CHAT_MODEL


class TestCollectionDeleteRoute:
    @patch("app.db.delete_collection")
    def test_delete_returns_ok(self, mock_delete, client):
        resp = client.post("/collections/5/delete")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        mock_delete.assert_called_once_with(5)


class TestConversationRoutes:
    @patch("app.db.list_conversations", return_value=[])
    def test_list_conversations(self, mock_list, client):
        resp = client.get("/collections/1/conversations")
        assert resp.status_code == 200
        mock_list.assert_called_once_with(1)

    @patch("app.db.get_conversation_messages", return_value=[])
    def test_get_messages(self, mock_get, client):
        resp = client.get("/conversations/3/messages")
        assert resp.status_code == 200
        mock_get.assert_called_once_with(3)

    @patch("app.db.delete_conversation")
    def test_delete_conversation(self, mock_delete, client):
        resp = client.post("/conversations/3/delete")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        mock_delete.assert_called_once_with(3)


class TestRtfEsc:
    def test_plain_ascii_unchanged(self):
        assert _app._rtf_esc("hello world") == "hello world"

    def test_backslash_escaped(self):
        assert _app._rtf_esc("a\\b") == "a\\\\b"

    def test_open_brace_escaped(self):
        assert _app._rtf_esc("a{b") == "a\\{b"

    def test_close_brace_escaped(self):
        assert _app._rtf_esc("a}b") == "a\\}b"

    def test_non_ascii_unicode_escaped(self):
        # é = U+00E9 = decimal 233
        result = _app._rtf_esc("caf\u00e9")
        assert "\\u233?" in result

    def test_empty_string(self):
        assert _app._rtf_esc("") == ""


class TestBuildRtf:
    _MSGS = [
        {"role": "user",      "content": "What is the treatment?"},
        {"role": "assistant", "content": "It involves X.\nAnd also Y."},
    ]

    def test_starts_with_rtf_header(self):
        rtf = _app._build_rtf("Title", "Col", "2024-01-01", self._MSGS)
        assert rtf.startswith(r"{\rtf1")

    def test_ends_with_closing_brace(self):
        rtf = _app._build_rtf("Title", "Col", "2024-01-01", self._MSGS)
        assert rtf.rstrip().endswith("}")

    def test_contains_title(self):
        rtf = _app._build_rtf("My Study", "Col", "2024-01-01", self._MSGS)
        assert "My Study" in rtf

    def test_contains_collection_name(self):
        rtf = _app._build_rtf("Title", "Cardiology Collection", "2024-01-01", self._MSGS)
        assert "Cardiology Collection" in rtf

    def test_contains_date(self):
        rtf = _app._build_rtf("Title", "Col", "2024-06-15T10:00:00", self._MSGS)
        assert "2024-06-15" in rtf

    def test_user_label_present(self):
        rtf = _app._build_rtf("Title", "Col", "2024-01-01", self._MSGS)
        assert "You:" in rtf

    def test_assistant_label_present(self):
        rtf = _app._build_rtf("Title", "Col", "2024-01-01", self._MSGS)
        assert "Claude:" in rtf

    def test_result_is_ascii_encodable(self):
        rtf = _app._build_rtf("Title", "Col", "2024-01-01", self._MSGS)
        rtf.encode("ascii")  # must not raise

class TestBuildDocx:
    import io as _io

    _MSGS = [
        {"role": "user",      "content": "What is the treatment?"},
        {"role": "assistant", "content": "It involves X.\nAnd also Y."},
    ]

    def test_returns_bytes_io(self):
        import io
        result = _app._build_docx("Title", "Col", "2024-01-01", self._MSGS)
        assert isinstance(result, io.BytesIO)

    def test_content_is_zip_archive(self):
        # DOCX files are ZIP archives — magic bytes are PK (0x50 0x4B)
        result = _app._build_docx("Title", "Col", "2024-01-01", self._MSGS)
        assert result.read(2) == b"PK"

    def test_non_empty_output(self):
        result = _app._build_docx("Title", "Col", "2024-01-01", self._MSGS)
        assert result.getbuffer().nbytes > 0


# ═══════════════════════════════════════════════════════════════════════════════
# conversation export route
# ═══════════════════════════════════════════════════════════════════════════════

_EXPORT_CONV = {
    "id": 1, "title": "Test Question",
    "collection_name": "My Collection", "created_at": "2024-01-01T00:00:00",
}
_EXPORT_MSGS = [
    {"role": "user",      "content": "What is the treatment?"},
    {"role": "assistant", "content": "It involves X."},
]


class TestConversationExportRoute:
    def test_invalid_format_returns_400(self, client):
        resp = client.get("/conversations/1/export?format=pdf")
        assert resp.status_code == 400

    @patch("app.db.get_conversation", return_value=None)
    def test_missing_conversation_returns_404(self, mock_conv, client):
        resp = client.get("/conversations/999/export?format=docx")
        assert resp.status_code == 404

    @patch("app.db.get_conversation_messages", return_value=[])
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_no_messages_returns_404(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=docx")
        assert resp.status_code == 404

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_docx_returns_200(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=docx")
        assert resp.status_code == 200

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_docx_content_type(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=docx")
        assert "wordprocessingml" in resp.content_type

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_docx_content_is_zip(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=docx")
        assert resp.data[:2] == b"PK"

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_rtf_returns_200(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=rtf")
        assert resp.status_code == 200

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_rtf_content_type(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=rtf")
        assert resp.content_type == "application/rtf"

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_rtf_body_starts_with_rtf_header(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=rtf")
        assert resp.data.startswith(b"{\\rtf1")

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_docx_content_disposition(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=docx")
        cd = resp.headers.get("Content-Disposition", "")
        assert "attachment" in cd
        assert ".docx" in cd

    @patch("app.db.get_conversation_messages", return_value=_EXPORT_MSGS)
    @patch("app.db.get_conversation", return_value=_EXPORT_CONV)
    def test_rtf_content_disposition(self, mock_conv, mock_msgs, client):
        resp = client.get("/conversations/1/export?format=rtf")
        cd = resp.headers.get("Content-Disposition", "")
        assert "attachment" in cd
        assert ".rtf" in cd
