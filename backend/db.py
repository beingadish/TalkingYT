from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from backend.config import Settings


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when DATABASE_URL is not set."""


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


def _schema_sql(embedding_dim: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        title      TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS videos (
        session_id            TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        video_id              TEXT NOT NULL,
        source_url            TEXT NOT NULL,
        transcript_characters INTEGER NOT NULL DEFAULT 0,
        snippet_count         INTEGER NOT NULL DEFAULT 0,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (session_id, video_id)
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id          BIGSERIAL PRIMARY KEY,
        session_id  TEXT NOT NULL,
        video_id    TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        timestamp   TEXT,
        source_url  TEXT,
        embedding   vector({embedding_dim}) NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        FOREIGN KEY (session_id, video_id)
            REFERENCES videos(session_id, video_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS chunks_session_idx ON chunks (session_id);
    CREATE INDEX IF NOT EXISTS chunks_session_video_idx ON chunks (session_id, video_id);

    CREATE TABLE IF NOT EXISTS messages (
        id         BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        sources    JSONB,
        evaluation JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS messages_session_idx ON messages (session_id, created_at);
    """


async def init_schema(settings: Settings) -> None:
    """Create the pgvector extension and all tables (idempotent)."""
    if not settings.database_url:
        raise DatabaseNotConfiguredError("DATABASE_URL is not configured.")
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute(_schema_sql(settings.embedding_dim))
        await conn.commit()


async def create_pool(settings: Settings) -> AsyncConnectionPool:
    """Open an async connection pool with the pgvector type registered."""
    if not settings.database_url:
        raise DatabaseNotConfiguredError("DATABASE_URL is not configured.")
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure_connection,
        open=False,
    )
    await pool.open(wait=True)
    return pool
