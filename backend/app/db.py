import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def resolve_sqlite_path(database_url: str) -> str:
    """Convert a SQLite DATABASE_URL into a filesystem path or ':memory:'."""
    if database_url == "sqlite:///:memory:":
        return ":memory:"
    if database_url.startswith("sqlite:////"):
        return "/" + database_url[len("sqlite:////") :]
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    return database_url


class Database:
    def __init__(self, database_url: str) -> None:
        self.db_path = resolve_sqlite_path(database_url)
        # Keep a persistent connection only when using in-memory SQLite
        self._memory_conn: Optional[sqlite3.Connection] = None

    def _ensure_parent_dir(self) -> None:
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent):
                os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        self._ensure_parent_dir()
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if self._memory_conn is None:
            conn.close()

    def init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id
                    ON messages (session_id)
                    """
                )
        finally:
            self._close(conn)

    def check_health(self) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("SELECT 1")
            row = cur.fetchone()
            return row is not None and row[0] == 1
        except Exception:
            return False
        finally:
            self._close(conn)

    def get_session_context(
        self, session_id: str
    ) -> Tuple[bool, Optional[str], List[Dict[str, str]]]:
        """Return (session_exists, stored_system_prompt, history_messages)."""
        conn = self._connect()
        try:
            sess_row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            msg_rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            exists = sess_row is not None or len(msg_rows) > 0
            stored_prompt = sess_row["system_prompt"] if sess_row is not None else None
            history = [
                {"role": row["role"], "content": row["content"]} for row in msg_rows
            ]
            return exists, stored_prompt, history
        finally:
            self._close(conn)

    def save_turn(
        self,
        session_id: str,
        system_prompt: Optional[str],
        user_content: str,
        assistant_content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Atomically persist the session, user message, and assistant message."""
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO sessions (session_id, system_prompt, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (session_id, system_prompt, now_iso),
                )
                if system_prompt is not None:
                    conn.execute(
                        "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
                        (system_prompt, session_id),
                    )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'user', ?, NULL, ?)
                    """,
                    (session_id, user_content, now_iso),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'assistant', ?, ?, ?)
                    """,
                    (session_id, assistant_content, metadata_json, now_iso),
                )
        finally:
            self._close(conn)

    def get_session_messages(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return oldest-first messages for a session, or None if session does not exist."""
        conn = self._connect()
        try:
            sess_row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            msg_rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            if sess_row is None and len(msg_rows) == 0:
                return None

            result: List[Dict[str, Any]] = []
            for row in msg_rows:
                item: Dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                }
                if row["role"] == "assistant" and row["metadata"]:
                    try:
                        item["metadata"] = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        item["metadata"] = {}
                result.append(item)
            return result
        finally:
            self._close(conn)

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
