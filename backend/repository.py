from __future__ import annotations

from datetime import datetime

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool


class SessionNotFoundError(KeyError):
    """Raised when a session id does not exist."""


class VideoNotFoundError(KeyError):
    """Raised when a video id does not exist within a session."""


class ChunkRecord:
    __slots__ = ("content", "video_id", "timestamp", "source_url", "score")

    def __init__(
        self,
        content: str,
        video_id: str,
        timestamp: str | None,
        source_url: str | None,
        score: float,
    ) -> None:
        self.content = content
        self.video_id = video_id
        self.timestamp = timestamp
        self.source_url = source_url
        self.score = score


class Repository:
    """Async data-access layer backed by PostgreSQL + pgvector."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def create_session(self, session_id: str, title: str | None) -> datetime:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO sessions (session_id, title) VALUES (%s, %s) "
                    "RETURNING created_at",
                    (session_id, title),
                )
                row = await cur.fetchone()
            await conn.commit()
        return row[0]

    async def session_exists(self, session_id: str) -> bool:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT 1 FROM sessions WHERE session_id = %s", (session_id,)
                )
                return await cur.fetchone() is not None

    async def count_sessions(self) -> int:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM sessions")
                row = await cur.fetchone()
        return int(row[0])

    async def add_video(
        self,
        session_id: str,
        video_id: str,
        source_url: str,
        transcript_characters: int,
        snippet_count: int,
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO videos "
                    "(session_id, video_id, source_url, transcript_characters, snippet_count) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (session_id, video_id) DO UPDATE SET "
                    "source_url = EXCLUDED.source_url, "
                    "transcript_characters = EXCLUDED.transcript_characters, "
                    "snippet_count = EXCLUDED.snippet_count",
                    (
                        session_id,
                        video_id,
                        source_url,
                        transcript_characters,
                        snippet_count,
                    ),
                )
            await conn.commit()

    async def add_chunks(
        self,
        session_id: str,
        video_id: str,
        chunks: list[dict],
    ) -> None:
        """Persist embedded chunks for a single video."""
        if not chunks:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks "
                    "(session_id, video_id, chunk_index, content, timestamp, source_url, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            session_id,
                            video_id,
                            chunk["chunk_index"],
                            chunk["content"],
                            chunk["timestamp"],
                            chunk["source_url"],
                            chunk["embedding"],
                        )
                        for chunk in chunks
                    ],
                )
            await conn.commit()

    async def remove_video(self, session_id: str, video_id: str) -> None:
        """Delete a single video; its chunks cascade away, others untouched."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM videos WHERE session_id = %s AND video_id = %s",
                    (session_id, video_id),
                )
                deleted = cur.rowcount
            await conn.commit()
        if not deleted:
            raise VideoNotFoundError(
                f"Video {video_id} was not found in session {session_id}."
            )

    async def delete_session(self, session_id: str) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM sessions WHERE session_id = %s", (session_id,)
                )
                deleted = cur.rowcount
            await conn.commit()
        if not deleted:
            raise SessionNotFoundError(f"Session {session_id} was not found.")

    async def get_session_summary(self, session_id: str) -> dict:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT session_id, title, created_at FROM sessions "
                    "WHERE session_id = %s",
                    (session_id,),
                )
                session = await cur.fetchone()
                if session is None:
                    raise SessionNotFoundError(f"Session {session_id} was not found.")
                summary = await self._summary_from_session(cur, session)
        return summary

    async def list_session_summaries(self) -> list[dict]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT session_id, title, created_at FROM sessions "
                    "ORDER BY created_at DESC"
                )
                sessions = await cur.fetchall()
                summaries = []
                for session in sessions:
                    summaries.append(await self._summary_from_session(cur, session))
        return summaries

    async def _summary_from_session(self, cur, session: dict) -> dict:
        session_id = session["session_id"]
        await cur.execute(
            "SELECT video_id, source_url, transcript_characters, snippet_count "
            "FROM videos WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        videos = await cur.fetchall()
        await cur.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE session_id = %s", (session_id,)
        )
        total_chunks = (await cur.fetchone())["c"]
        total_chars = sum(v["transcript_characters"] for v in videos)
        return {
            "session_id": session_id,
            "title": session["title"],
            "created_at": session["created_at"],
            "videos": videos,
            "total_chunks": total_chunks,
            "total_transcript_characters": total_chars,
        }

    async def similarity_search(
        self,
        session_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ChunkRecord]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT content, video_id, timestamp, source_url, "
                    "1 - (embedding <=> %s) AS score "
                    "FROM chunks WHERE session_id = %s "
                    "ORDER BY embedding <=> %s ASC LIMIT %s",
                    (query_embedding, session_id, query_embedding, top_k),
                )
                rows = await cur.fetchall()
        return [
            ChunkRecord(
                content=row["content"],
                video_id=row["video_id"],
                timestamp=row["timestamp"],
                source_url=row["source_url"],
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list | None = None,
        evaluation: dict | None = None,
    ) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO messages (session_id, role, content, sources, evaluation) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        session_id,
                        role,
                        content,
                        Jsonb(sources) if sources is not None else None,
                        Jsonb(evaluation) if evaluation is not None else None,
                    ),
                )
            await conn.commit()

    async def get_messages(self, session_id: str) -> list[dict]:
        async with self.pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT role, content, sources, evaluation, created_at "
                    "FROM messages WHERE session_id = %s ORDER BY created_at ASC, id ASC",
                    (session_id,),
                )
                rows = await cur.fetchall()
        return rows
