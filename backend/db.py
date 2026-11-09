"""SQLite storage: cases (full JSON), audit log, and model-call metrics for the A/B comparison."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .schemas import Case, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS case_seq (n INTEGER PRIMARY KEY AUTOINCREMENT);
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, time TEXT NOT NULL,
    actor TEXT NOT NULL, action TEXT NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, time TEXT NOT NULL, task TEXT NOT NULL,
    model TEXT NOT NULL, input_tokens INTEGER, output_tokens INTEGER, seconds REAL, valid INTEGER);
"""


class Database:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(_SCHEMA)

    def next_case_id(self) -> str:
        with self.lock:
            n = self.conn.execute("INSERT INTO case_seq DEFAULT VALUES").lastrowid
            self.conn.commit()
        return f"FC-{n:04d}"

    def save_case(self, case: Case) -> None:
        now = now_iso()
        with self.lock:
            self.conn.execute(
                "INSERT INTO cases (case_id, status, data, created_at, updated_at) VALUES (?,?,?,?,?) "
                "ON CONFLICT(case_id) DO UPDATE SET status=excluded.status, data=excluded.data, "
                "updated_at=excluded.updated_at",
                (case.case_id, case.status, case.model_dump_json(), case.created_at, now))
            self.conn.commit()

    def get_case(self, case_id: str) -> Case | None:
        with self.lock:
            row = self.conn.execute("SELECT data FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return Case.model_validate_json(row["data"]) if row else None

    def list_cases(self, limit: int = 50) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT case_id, status, created_at, updated_at FROM cases ORDER BY created_at DESC, case_id DESC "
                "LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def audit(self, case_id: str | None, actor: str, action: str, detail: dict | None = None) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO audit_log (case_id, time, actor, action, detail) VALUES (?,?,?,?,?)",
                              (case_id, now_iso(), actor, action, json.dumps(detail or {}, ensure_ascii=False)))
            self.conn.commit()

    def get_audit(self, case_id: str) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT time, actor, action, detail FROM audit_log WHERE case_id=? ORDER BY id",
                                     (case_id,)).fetchall()
        return [{**dict(r), "detail": json.loads(r["detail"] or "{}")} for r in rows]

    def log_call(self, case_id: str | None, task: str, model: str, input_tokens: int, output_tokens: int,
                 seconds: float, valid: bool) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO model_calls (case_id, time, task, model, input_tokens, output_tokens, seconds, valid) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (case_id, now_iso(), task, model, input_tokens, output_tokens, seconds, int(valid)))
            self.conn.commit()

    def call_stats(self) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT model, task, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, "
                "SUM(output_tokens) AS output_tokens, ROUND(AVG(seconds), 3) AS avg_seconds, "
                "SUM(valid) AS valid_calls FROM model_calls GROUP BY model, task ORDER BY model, task").fetchall()
        return [dict(r) for r in rows]
