
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "./knowledge.db")


@contextmanager
def get_db():
    """Open a database connection and manage transactions."""
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the database tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL
                    CHECK(source_type IN ('note', 'url')),
                title TEXT NOT NULL,
                source_url TEXT,
                raw_content TEXT NOT NULL,
                created_at TEXT NOT NULL
                    DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,

                FOREIGN KEY (item_id)
                    REFERENCES items(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_item_id
                ON chunks(item_id);
        """)
