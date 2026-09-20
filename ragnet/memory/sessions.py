"""Persistent conversation memory (SQLite, stdlib only)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class Sessions:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 session_id TEXT NOT NULL,
                 role TEXT NOT NULL,
                 content TEXT NOT NULL,
                 ts REAL NOT NULL)"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)")
        self.conn.commit()

    def add(self, session_id: str, role: str, content: str) -> None:
        self.conn.execute("INSERT INTO messages(session_id, role, content, ts) VALUES (?,?,?,?)", (session_id, role, content, time.time()))
        self.conn.commit()

    def history(self, session_id: str, turns: int = 6) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?", (session_id, turns * 2)
        ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def clear(self, session_id: str) -> None:
        self.conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        self.conn.commit()

    def list(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT session_id, COUNT(*), MAX(ts) FROM messages GROUP BY session_id ORDER BY MAX(ts) DESC"
        ).fetchall()
        return [{"session_id": s, "messages": n, "last_ts": t} for s, n, t in rows]
