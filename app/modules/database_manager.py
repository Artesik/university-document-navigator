"""SQLite persistence and JSON knowledge-base import."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .documents import canonical, demo_document, load_knowledge


class DatabaseManager:
    def __init__(self, db_path: str | Path, knowledge_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.knowledge_path = Path(knowledge_path) if knowledge_path else None

    @contextmanager
    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('PRAGMA foreign_keys=ON')
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create runtime tables and synchronize public demo knowledge from JSON."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY,
                    category TEXT NOT NULL,
                    question TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    department TEXT NOT NULL,
                    keywords_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    knowledge_id INTEGER,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL,
                    helpful INTEGER NOT NULL,
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    format TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    pages_json TEXT NOT NULL,
                    raw BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
        if self.knowledge_path:
            self.import_knowledge_base()

    def import_knowledge_base(self) -> None:
        items = load_knowledge(self.knowledge_path)
        with self._connect() as connection:
            connection.execute("DELETE FROM knowledge")
            connection.executemany(
                """
                INSERT INTO knowledge
                (id, category, question, aliases_json, answer, steps_json, source, department, keywords_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["id"], item["category"], item["question"],
                        json.dumps(item.get("aliases", []), ensure_ascii=False),
                        item["answer"], json.dumps(item.get("steps", []), ensure_ascii=False),
                        item["source"], item["department"],
                        json.dumps(item.get("keywords", []), ensure_ascii=False),
                    )
                    for item in items
                ],
            )

    def import_demo_documents(self, path):
        documents = [demo_document(item) for item in load_knowledge(path)]
        with self._connect() as connection:
            for document in documents:
                self._save_document(connection, document)
        return len(documents)

    @staticmethod
    def _save_document(connection, document):
        connection.execute(
            '''INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, format=excluded.format,
               sha256=excluded.sha256, metadata_json=excluded.metadata_json,
               pages_json=excluded.pages_json, raw=excluded.raw''',
            (document['id'], document['name'], document['format'], document['sha256'],
             canonical(document['metadata']), canonical(document['pages']), document['raw'], _now()),
        )

    def add_document(self, document):
        with self._connect() as connection:
            self._save_document(connection, document)

    def get_documents(self):
        with self._connect() as connection:
            rows = connection.execute('SELECT id,name,format,sha256,metadata_json,pages_json FROM documents ORDER BY id').fetchall()
        result = []
        for row in rows:
            doc = dict(row)
            doc['metadata'] = json.loads(doc.pop('metadata_json'))
            doc['pages'] = json.loads(doc.pop('pages_json'))
            result.append(doc)
        return result

    def get_original(self, document_id):
        with self._connect() as connection:
            row = connection.execute('SELECT raw FROM documents WHERE id=?', (document_id,)).fetchone()
        return bytes(row['raw']) if row else None

    def delete_document(self, document_id):
        with self._connect() as connection:
            connection.execute('DELETE FROM documents WHERE id=?', (document_id,))

    def get_knowledge_items(self, category: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM knowledge"
        params: tuple[Any, ...] = ()
        if category and category != "Все категории":
            sql += " WHERE category = ?"
            params = (category,)
        sql += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._decode_item(row) for row in rows]

    def get_categories(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT DISTINCT category FROM knowledge ORDER BY category").fetchall()
        return [row[0] for row in rows]

    def add_history(self, query: str, knowledge_id: int | None, score: float) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO history (query, knowledge_id, score, created_at) VALUES (?, ?, ?, ?)",
                (query, knowledge_id, float(score), _now()),
            )
            return int(cursor.lastrowid)

    def get_history(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT h.id, h.query, h.knowledge_id, h.score, h.created_at, k.question, k.category
                FROM history h LEFT JOIN knowledge k ON k.id = h.knowledge_id
                ORDER BY h.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_history(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM feedback")
            connection.execute("DELETE FROM history")

    def add_feedback(self, history_id: int, helpful: bool, comment: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback (history_id, helpful, comment, created_at) VALUES (?, ?, ?, ?)",
                (history_id, int(helpful), comment.strip(), _now()),
            )

    @staticmethod
    def _decode_item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["aliases"] = json.loads(item.pop("aliases_json"))
        item["steps"] = json.loads(item.pop("steps_json"))
        item["keywords"] = json.loads(item.pop("keywords_json"))
        return item


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
