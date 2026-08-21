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
Unit tests for db.py — all external I/O (psycopg2, pgvector) is mocked.
Run with:  pytest tests/test_db.py -v
"""
import json
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

import db


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_conn(rows=None, rowcount=1):
    """Return a mock psycopg2 connection whose cursor yields *rows* on fetchall/fetchone."""
    cur = MagicMock()
    cur.fetchone.return_value = rows[0] if rows else None
    cur.fetchall.return_value = rows if rows else []
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


@pytest.fixture
def mock_conn():
    """Patch db._connect so db_conn() never touches Postgres."""
    conn, cur = _make_conn()
    with patch("db._connect", return_value=conn):
        yield conn, cur


# ── db_conn context manager ───────────────────────────────────────────────────

def _raise_inside_db_conn(exc: Exception):
    with db.db_conn():
        raise exc


class TestDbConn:
    def test_commits_on_success(self):
        conn, _ = _make_conn()
        with patch("db._connect", return_value=conn):
            with db.db_conn() as c:
                assert c is conn
        conn.commit.assert_called_once()
        conn.rollback.assert_not_called()

    def test_rollback_on_exception(self):
        conn, _ = _make_conn()
        with patch("db._connect", return_value=conn):
            with pytest.raises(ValueError):
                _raise_inside_db_conn(ValueError("boom"))
        conn.rollback.assert_called_once()
        conn.commit.assert_not_called()

    def test_connection_always_closed(self):
        conn, _ = _make_conn()
        with patch("db._connect", return_value=conn):
            with pytest.raises(RuntimeError):
                _raise_inside_db_conn(RuntimeError("err"))
        conn.close.assert_called_once()

    def test_connection_closed_on_success_too(self):
        conn, _ = _make_conn()
        with patch("db._connect", return_value=conn):
            with db.db_conn() as c:
                assert c is conn
        conn.close.assert_called_once()


# ── create_collection ─────────────────────────────────────────────────────────

class TestCreateCollection:
    def test_returns_new_id(self):
        conn, _ = _make_conn(rows=[(42,)])
        with patch("db._connect", return_value=conn):
            cid = db.create_collection("My Collection", "diabetes", "diabetes[MeSH]", 7)
        assert cid == 42

    def test_inserts_correct_values(self):
        conn, cur = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            db.create_collection("Name", "user q", "pubmed q", 7)
        sql, params = cur.execute.call_args.args
        assert "INSERT INTO rag_collections" in sql
        assert params == ("Name", "user q", "pubmed q", 7)


# ── add_article ───────────────────────────────────────────────────────────────

class TestAddArticle:
    _ART = {
        "pmid": "999", "title": "T", "authors": "A", "journal": "J",
        "year": "2024", "abstract": "Abs", "url": "http://x",
    }

    def test_inserts_all_fields(self, mock_conn):
        _, cur = mock_conn
        db.add_article(7, self._ART, has_full_text=True, pmcid="PMC123")
        sql, params = cur.execute.call_args.args
        assert "INSERT INTO rag_articles" in sql
        assert params[0] == 7           # collection_id
        assert params[1] == "999"       # pmid
        assert params[8] is True        # has_full_text
        assert params[9] == "PMC123"    # pmcid

    def test_defaults_has_full_text_false(self, mock_conn):
        _, cur = mock_conn
        db.add_article(1, self._ART)
        _, params = cur.execute.call_args.args
        assert params[8] is False

    def test_defaults_pmcid_none(self, mock_conn):
        _, cur = mock_conn
        db.add_article(1, self._ART)
        _, params = cur.execute.call_args.args
        assert params[9] is None


# ── chunks_exist ──────────────────────────────────────────────────────────────

class TestChunksExist:
    def test_returns_true_when_row_found(self):
        conn, _ = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            assert db.chunks_exist("111") is True

    def test_returns_false_when_no_row(self):
        conn, cur = _make_conn(rows=[])
        cur.fetchone.return_value = None
        with patch("db._connect", return_value=conn):
            assert db.chunks_exist("111") is False

    def test_queries_correct_pmid(self):
        conn, cur = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            db.chunks_exist("pmid-xyz")
        sql, params = cur.execute.call_args.args
        assert "article_chunks" in sql
        assert params == ("pmid-xyz",)


# ── save_chunks ───────────────────────────────────────────────────────────────

class TestSaveChunks:
    def test_calls_execute_values(self, mock_conn):
        _, _ = mock_conn
        chunks = ["chunk one", "chunk two"]
        embs = [np.zeros(384), np.ones(384)]
        with patch("psycopg2.extras.execute_values") as mock_ev:
            db.save_chunks("pmid-1", chunks, embs)
        mock_ev.assert_called_once()

    def test_data_indexed_correctly(self, mock_conn):
        _, _ = mock_conn
        chunks = ["alpha", "beta"]
        embs = [np.array([0.1] * 384), np.array([0.2] * 384)]
        with patch("psycopg2.extras.execute_values") as mock_ev:
            db.save_chunks("pmid-2", chunks, embs)
        _, data = mock_ev.call_args.args[1], mock_ev.call_args.args[2]
        assert data[0] == ("pmid-2", 0, "alpha", embs[0])
        assert data[1] == ("pmid-2", 1, "beta", embs[1])

    def test_on_conflict_in_sql(self, mock_conn):
        _, _ = mock_conn
        with patch("psycopg2.extras.execute_values") as mock_ev:
            db.save_chunks("p", ["c"], [np.zeros(384)])
        sql = mock_ev.call_args.args[1]
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    def test_empty_chunks_still_calls_execute_values(self, mock_conn):
        _, _ = mock_conn
        with patch("psycopg2.extras.execute_values") as mock_ev:
            db.save_chunks("p", [], [])
        mock_ev.assert_called_once()


# ── list_collections ──────────────────────────────────────────────────────────

class TestListCollections:
    def _row(self, id=1, name="C", user_query="q", pubmed_query="pq",
             created_at="2024-01-01", article_count=2, full_text_count=1, chunk_count=10):
        r = MagicMock()
        r.keys.return_value = [
            "id", "name", "user_query", "pubmed_query", "created_at",
            "article_count", "full_text_count", "chunk_count",
        ]
        r.__iter__ = lambda s: iter(r.keys())
        r.items.return_value = list({
            "id": id, "name": name, "user_query": user_query,
            "pubmed_query": pubmed_query, "created_at": created_at,
            "article_count": article_count, "full_text_count": full_text_count,
            "chunk_count": chunk_count,
        }.items())
        # dict(r) calls items()
        return r

    def test_returns_list_of_dicts(self):
        row = {"id": 1, "name": "C", "user_query": "q", "pubmed_query": "pq",
               "created_at": "2024-01-01", "article_count": 2,
               "full_text_count": 1, "chunk_count": 10}
        conn, cur = _make_conn()
        cur.fetchall.return_value = [row]
        with patch("db._connect", return_value=conn):
            result = db.list_collections(7)
        assert isinstance(result, list)
        assert result[0]["name"] == "C"

    def test_empty_returns_empty_list(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            result = db.list_collections(7)
        assert result == []


# ── get_collection ────────────────────────────────────────────────────────────

class TestGetCollection:
    def test_returns_dict_when_found(self):
        row = {"id": 5, "name": "X", "user_query": "u", "pubmed_query": "p",
               "created_at": "2024-01-01", "article_count": 0,
               "full_text_count": 0, "chunk_count": 0}
        conn, cur = _make_conn()
        cur.fetchone.return_value = row
        with patch("db._connect", return_value=conn):
            result = db.get_collection(5, 7)
        assert result["id"] == 5

    def test_returns_none_when_not_found(self):
        conn, cur = _make_conn()
        cur.fetchone.return_value = None
        with patch("db._connect", return_value=conn):
            result = db.get_collection(999, 7)
        assert result is None

    def test_passes_cid_to_query(self):
        conn, cur = _make_conn()
        cur.fetchone.return_value = None
        with patch("db._connect", return_value=conn):
            db.get_collection(77, 7)
        _, params = cur.execute.call_args.args
        assert params == (77, 7)


# ── get_collection_articles ───────────────────────────────────────────────────

class TestGetCollectionArticles:
    def test_returns_list(self):
        row = {"id": 1, "pmid": "1", "title": "T", "authors": "A",
               "journal": "J", "year": "2024", "abstract": "ab",
               "url": "u", "has_full_text": False, "pmcid": None}
        conn, cur = _make_conn()
        cur.fetchall.return_value = [row]
        with patch("db._connect", return_value=conn):
            rows = db.get_collection_articles(3)
        assert len(rows) == 1
        assert rows[0]["pmid"] == "1"

    def test_filters_by_collection_id(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            db.get_collection_articles(12)
        _, params = cur.execute.call_args.args
        assert params == (12,)


# ── semantic_search ───────────────────────────────────────────────────────────

class TestSemanticSearch:
    def _chunk_row(self, pmid="1", similarity="0.850"):
        return {
            "pmid": pmid, "title": "T", "authors": "A", "journal": "J",
            "year": "2024", "abstract": "abs", "url": "u",
            "has_full_text": False, "chunk_text": "text",
            "chunk_index": 0, "similarity": similarity,
        }

    def test_similarity_cast_to_float(self):
        row = self._chunk_row(similarity="0.850")
        conn, cur = _make_conn()
        cur.fetchall.return_value = [row]
        with patch("db._connect", return_value=conn):
            results = db.semantic_search(1, np.zeros(384), k=1)
        assert isinstance(results[0]["similarity"], float)
        assert results[0]["similarity"] == pytest.approx(0.850)

    def test_empty_results_returns_empty_list(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            results = db.semantic_search(1, np.zeros(384), k=5)
        assert results == []

    def test_passes_embedding_and_cid(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        emb = np.ones(384)
        with patch("db._connect", return_value=conn):
            db.semantic_search(7, emb, k=3)
        _, params = cur.execute.call_args.args
        # params: (embedding, cid, embedding, k)
        assert params[1] == 7
        assert params[3] == 3
        np.testing.assert_array_equal(params[0], emb)

    def test_multiple_rows_all_similarities_float(self):
        rows = [self._chunk_row("1", "0.9"), self._chunk_row("2", "0.7")]
        conn, cur = _make_conn()
        cur.fetchall.return_value = rows
        with patch("db._connect", return_value=conn):
            results = db.semantic_search(1, np.zeros(384))
        for r in results:
            assert isinstance(r["similarity"], float)


# ── delete_collection ─────────────────────────────────────────────────────────

class TestDeleteCollection:
    def test_issues_delete(self, mock_conn):
        _, cur = mock_conn
        db.delete_collection(5, 7)
        sql, params = cur.execute.call_args.args
        assert "DELETE FROM rag_collections" in sql
        assert params == (5, 7)


# ── create_conversation ───────────────────────────────────────────────────────

class TestCreateConversation:
    def test_returns_new_id(self):
        conn, _ = _make_conn(rows=[(99,)])
        with patch("db._connect", return_value=conn):
            vid = db.create_conversation(3, "Chat title")
        assert vid == 99

    def test_title_truncated_to_120_chars(self):
        long_title = "x" * 200
        conn, cur = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            db.create_conversation(1, long_title)
        _, params = cur.execute.call_args.args
        assert len(params[1]) == 120

    def test_short_title_not_truncated(self):
        conn, cur = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            db.create_conversation(1, "short")
        _, params = cur.execute.call_args.args
        assert params[1] == "short"

    def test_title_exactly_120_chars_unchanged(self):
        title = "y" * 120
        conn, cur = _make_conn(rows=[(1,)])
        with patch("db._connect", return_value=conn):
            db.create_conversation(1, title)
        _, params = cur.execute.call_args.args
        assert params[1] == title


# ── add_message ───────────────────────────────────────────────────────────────

class TestAddMessage:
    def test_inserts_without_citations(self, mock_conn):
        _, cur = mock_conn
        db.add_message(10, "user", "Hello")
        sql, params = cur.execute.call_args.args
        assert "INSERT INTO conversation_messages" in sql
        assert params[0] == 10
        assert params[1] == "user"
        assert params[2] == "Hello"
        assert params[3] is None

    def test_inserts_with_citations(self, mock_conn):
        _, cur = mock_conn
        citations = [{"pmid": "123", "title": "T"}]
        db.add_message(10, "assistant", "Answer", citations=citations)
        _, params = cur.execute.call_args.args
        # citations wrapped in psycopg2.extras.Json — check it's not None
        assert params[3] is not None

    def test_user_role(self, mock_conn):
        _, cur = mock_conn
        db.add_message(1, "user", "content")
        _, params = cur.execute.call_args.args
        assert params[1] == "user"

    def test_assistant_role(self, mock_conn):
        _, cur = mock_conn
        db.add_message(1, "assistant", "content")
        _, params = cur.execute.call_args.args
        assert params[1] == "assistant"


# ── list_conversations ────────────────────────────────────────────────────────

class TestListConversations:
    def test_returns_list_for_collection(self):
        row = {"id": 1, "title": "Chat 1", "created_at": "2024-01-01", "message_count": 3}
        conn, cur = _make_conn()
        cur.fetchall.return_value = [row]
        with patch("db._connect", return_value=conn):
            result = db.list_conversations(5)
        assert len(result) == 1
        assert result[0]["title"] == "Chat 1"

    def test_filters_by_collection_id(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            db.list_conversations(42)
        _, params = cur.execute.call_args.args
        assert params == (42,)

    def test_empty_collection_returns_empty_list(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            assert db.list_conversations(1) == []


# ── get_conversation ──────────────────────────────────────────────────────────

class TestGetConversation:
    def test_returns_dict_when_found(self):
        row = {"id": 3, "title": "Chat", "created_at": "2024-01-01", "collection_name": "Col"}
        conn, cur = _make_conn()
        cur.fetchone.return_value = row
        with patch("db._connect", return_value=conn):
            result = db.get_conversation(3)
        assert result["id"] == 3
        assert result["collection_name"] == "Col"

    def test_returns_none_when_not_found(self):
        conn, cur = _make_conn()
        cur.fetchone.return_value = None
        with patch("db._connect", return_value=conn):
            assert db.get_conversation(999) is None

    def test_passes_vid_to_query(self):
        conn, cur = _make_conn()
        cur.fetchone.return_value = None
        with patch("db._connect", return_value=conn):
            db.get_conversation(55)
        _, params = cur.execute.call_args.args
        assert params == (55,)


# ── get_conversation_messages ─────────────────────────────────────────────────

class TestGetConversationMessages:
    def test_returns_ordered_messages(self):
        rows = [
            {"id": 1, "role": "user", "content": "Hi", "citations": None, "created_at": "2024-01-01"},
            {"id": 2, "role": "assistant", "content": "Hello", "citations": None, "created_at": "2024-01-01"},
        ]
        conn, cur = _make_conn()
        cur.fetchall.return_value = rows
        with patch("db._connect", return_value=conn):
            result = db.get_conversation_messages(7)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_filters_by_conversation_id(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            db.get_conversation_messages(88)
        _, params = cur.execute.call_args.args
        assert params == (88,)

    def test_empty_conversation_returns_empty_list(self):
        conn, cur = _make_conn()
        cur.fetchall.return_value = []
        with patch("db._connect", return_value=conn):
            assert db.get_conversation_messages(1) == []


# ── delete_conversation ───────────────────────────────────────────────────────

class TestDeleteConversation:
    def test_issues_delete(self, mock_conn):
        _, cur = mock_conn
        db.delete_conversation(13)
        sql, params = cur.execute.call_args.args
        assert "DELETE FROM conversations" in sql
        assert params == (13,)
