# MIT License
# Copyright (c) 2025 Anton Schreiner

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path

TMP_PATH = Path(__file__).parent / ".tmp"
TMP_PATH.mkdir(exist_ok=True)

DB_PATH = TMP_PATH / "graphs.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            data TEXT NOT NULL,
            parent_hash TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON graphs(hash)")
    conn.commit()
    conn.close()


def generate_hash(data: dict) -> str:
    """Generate a short hash from graph data + timestamp for uniqueness."""
    content = json.dumps(data, sort_keys=True) + str(datetime.utcnow().timestamp())
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def save_graph(name: str, data: dict, parent_hash: str = None) -> str:
    """Save a new graph and return its hash."""
    conn = get_connection()
    graph_hash = generate_hash(data)
    now = datetime.utcnow().isoformat()
    
    conn.execute(
        """INSERT INTO graphs (hash, name, data, parent_hash, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (graph_hash, name, json.dumps(data), parent_hash, now, now)
    )
    conn.commit()
    conn.close()
    return graph_hash


def update_graph(graph_hash: str, data: dict, name: str = None) -> bool:
    """Update an existing graph."""
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    
    if name:
        conn.execute(
            "UPDATE graphs SET data = ?, name = ?, updated_at = ? WHERE hash = ?",
            (json.dumps(data), name, now, graph_hash)
        )
    else:
        conn.execute(
            "UPDATE graphs SET data = ?, updated_at = ? WHERE hash = ?",
            (json.dumps(data), now, graph_hash)
        )
    
    affected = conn.total_changes
    conn.commit()
    conn.close()
    return affected > 0


def get_graph(graph_hash: str) -> dict | None:
    """Retrieve a graph by its hash."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM graphs WHERE hash = ?", (graph_hash,)
    ).fetchone()
    conn.close()
    
    if row:
        return {
            "hash": row["hash"],
            "name": row["name"],
            "data": json.loads(row["data"]),
            "parent_hash": row["parent_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
    return None


def list_graphs(limit: int = 50, offset: int = 0) -> list[dict]:
    """List all graphs with pagination."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT hash, name, parent_hash, created_at, updated_at 
           FROM graphs ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
        (limit, offset)
    ).fetchall()
    conn.close()
    
    return [
        {
            "hash": row["hash"],
            "name": row["name"],
            "parent_hash": row["parent_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
        for row in rows
    ]


def delete_graph(graph_hash: str) -> bool:
    """Delete a graph by its hash."""
    conn = get_connection()
    conn.execute("DELETE FROM graphs WHERE hash = ?", (graph_hash,))
    affected = conn.total_changes
    conn.commit()
    conn.close()
    return affected > 0


# Initialize database on import
init_db()
