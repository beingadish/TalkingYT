from __future__ import annotations

import asyncio
import re
from uuid import uuid4

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility with older LangChain installs.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import Settings
from backend.evaluation import EvaluationPayload, RagasEvaluator
from backend.models import (
    ChatResponse,
    IndexResponse,
    MessageItem,
    RagasEvaluation,
    SessionSummary,
    SourceChunk,
    VideoSummary,
)
from backend.repository import (
    ChunkRecord,
    Repository,
    SessionNotFoundError,
    VideoNotFoundError,
)
from backend.transcripts import (
    TranscriptBundle,
    TranscriptError,
    fetch_transcript,
    source_url,
)


TIMESTAMP_RE = re.compile(r"\[(?P<timestamp>\d{2}:\d{2}(?::\d{2})?)\]")


class MissingConfigurationError(RuntimeError):
    """Raised when the configured model provider cannot be used."""


class ProviderError(RuntimeError):
    """Raised when the upstream model provider (Gemini) rejects a request."""


class RagEngine:
    def __init__(self, settings: Settings, repository: Repository):
        self.settings = settings
        self.repository = repository
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None
        self._llm: ChatGoogleGenerativeAI | None = None
        self._evaluator = RagasEvaluator(settings)

    async def create_session(
        self,
        videos: list[str],
        languages: list[str],
        title: str | None = None,
    ) -> IndexResponse:
        self._ensure_google_key()
        session_id = uuid4().hex[:12]
        await self.repository.create_session(session_id, title)
        indexed, total_chunks = await self._index_videos(session_id, videos, languages)
        summary = await self.repository.get_session_summary(session_id)
        return IndexResponse(
            **_summary_payload(summary),
            status="indexed",
            message=f"Indexed {indexed} video(s) into {total_chunks} transcript chunks.",
        )

    async def add_videos(
        self,
        session_id: str,
        videos: list[str],
        languages: list[str],
    ) -> IndexResponse:
        self._ensure_google_key()
        if not await self.repository.session_exists(session_id):
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        indexed, total_chunks = await self._index_videos(session_id, videos, languages)
        summary = await self.repository.get_session_summary(session_id)
        return IndexResponse(
            **_summary_payload(summary),
            status="indexed",
            message=f"Added {indexed} video(s) as {total_chunks} new transcript chunks.",
        )

    async def _index_videos(
        self,
        session_id: str,
        videos: list[str],
        languages: list[str],
    ) -> tuple[int, int]:
        bundles = await asyncio.gather(
            *(asyncio.to_thread(fetch_transcript, video, languages) for video in videos)
        )

        indexed = 0
        new_chunks = 0
        for bundle in bundles:
            documents = self._documents_from_transcript(bundle)
            if not documents:
                continue
            vectors = await asyncio.to_thread(
                self._embed_documents,
                [doc.page_content for doc in documents],
            )
            chunk_rows = [
                {
                    "chunk_index": doc.metadata["chunk_index"],
                    "content": doc.page_content,
                    "timestamp": doc.metadata.get("timestamp"),
                    "source_url": doc.metadata.get("source_url"),
                    "embedding": vector,
                }
                for doc, vector in zip(documents, vectors)
            ]
            await self.repository.add_video(
                session_id=session_id,
                video_id=bundle.video_id,
                source_url=bundle.source_url,
                transcript_characters=len(bundle.transcript_text),
                snippet_count=len(bundle.snippets),
            )
            await self.repository.add_chunks(session_id, bundle.video_id, chunk_rows)
            indexed += 1
            new_chunks += len(chunk_rows)

        if indexed == 0:
            raise TranscriptError("No transcript chunks were created.")
        return indexed, new_chunks

    async def remove_video(self, session_id: str, video_id: str) -> SessionSummary:
        if not await self.repository.session_exists(session_id):
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        await self.repository.remove_video(session_id, video_id)
        summary = await self.repository.get_session_summary(session_id)
        return SessionSummary(**_summary_payload(summary))

    async def chat(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
        evaluate: bool = True,
    ) -> ChatResponse:
        self._ensure_google_key()
        if not await self.repository.session_exists(session_id):
            raise SessionNotFoundError(f"Session {session_id} was not found.")

        retrieval_count = top_k or self.settings.default_top_k
        query_vector = await asyncio.to_thread(self._embed_query, question)
        matches = await self.repository.similarity_search(
            session_id, query_vector, retrieval_count
        )
        context = _format_context(matches)
        prompt = _build_prompt(question=question, context=context)

        response = await self._invoke_llm(prompt)
        sources = [_source_from_match(match) for match in matches]
        evaluation = (
            await self._evaluator.answer_relevancy(
                EvaluationPayload(
                    question=question,
                    answer=response,
                    contexts=[match.content for match in matches],
                )
            )
            if evaluate
            else RagasEvaluation(status="skipped", reason="Evaluation was disabled.")
        )

        await self.repository.add_message(session_id, "user", question)
        await self.repository.add_message(
            session_id,
            "assistant",
            response,
            sources=[src.model_dump() for src in sources],
            evaluation=evaluation.model_dump(),
        )

        return ChatResponse(
            session_id=session_id,
            answer=response,
            sources=sources,
            evaluation=evaluation,
        )

    async def list_sessions(self) -> list[SessionSummary]:
        summaries = await self.repository.list_session_summaries()
        return [SessionSummary(**_summary_payload(item)) for item in summaries]

    async def get_session(self, session_id: str) -> SessionSummary:
        summary = await self.repository.get_session_summary(session_id)
        return SessionSummary(**_summary_payload(summary))

    async def delete_session(self, session_id: str) -> None:
        await self.repository.delete_session(session_id)

    async def get_messages(self, session_id: str) -> list[MessageItem]:
        if not await self.repository.session_exists(session_id):
            raise SessionNotFoundError(f"Session {session_id} was not found.")
        rows = await self.repository.get_messages(session_id)
        return [
            MessageItem(
                role=row["role"],
                content=row["content"],
                sources=[SourceChunk(**src) for src in (row["sources"] or [])],
                evaluation=(
                    RagasEvaluation(**row["evaluation"]) if row["evaluation"] else None
                ),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _documents_from_transcript(self, bundle: TranscriptBundle) -> list[Document]:
        timed_transcript = "\n".join(
            f"[{format_timestamp(snippet.start)}] {snippet.text}"
            for snippet in bundle.snippets
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        documents = splitter.create_documents(
            [timed_transcript],
            metadatas=[
                {
                    "video_id": bundle.video_id,
                    "source_url": bundle.source_url,
                }
            ],
        )
        for index, doc in enumerate(documents):
            doc.metadata["chunk_index"] = index
            doc.metadata["timestamp"] = _first_timestamp(doc.page_content)
            doc.metadata["source_url"] = source_url(
                bundle.video_id,
                _timestamp_to_seconds(doc.metadata["timestamp"]),
            )
        return documents

    def _ensure_google_key(self) -> None:
        if not self.settings.google_api_key:
            raise MissingConfigurationError("GOOGLE_API_KEY is not configured.")

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._get_embeddings().embed_documents(texts)
        except Exception as exc:
            raise _as_provider_error(exc) from exc

    def _embed_query(self, text: str) -> list[float]:
        try:
            return self._get_embeddings().embed_query(text)
        except Exception as exc:
            raise _as_provider_error(exc) from exc

    def _get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.settings.google_embedding_model,
                google_api_key=self.settings.google_api_key,
            )
        return self._embeddings

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.settings.google_chat_model,
                google_api_key=self.settings.google_api_key,
                temperature=0.2,
            )
        return self._llm

    async def _invoke_llm(self, prompt: str) -> str:
        llm = self._get_llm()
        try:
            result = await llm.ainvoke(prompt)
        except AttributeError:
            result = await asyncio.to_thread(llm.invoke, prompt)
        except Exception as exc:
            raise _as_provider_error(exc) from exc
        content = getattr(result, "content", result)
        return str(content).strip()


def _summary_payload(summary: dict) -> dict:
    return {
        "session_id": summary["session_id"],
        "title": summary["title"],
        "created_at": summary["created_at"],
        "videos": [VideoSummary(**video) for video in summary["videos"]],
        "total_chunks": summary["total_chunks"],
        "total_transcript_characters": summary["total_transcript_characters"],
    }


def _as_provider_error(exc: Exception) -> Exception:
    """Translate upstream Gemini/provider failures into a clean ProviderError.

    Auth/invalid-key failures are the most common cause, so they are given a
    clear, actionable message. Already-handled domain errors pass through.
    """
    if isinstance(exc, (MissingConfigurationError, ProviderError, TranscriptError)):
        return exc
    message = str(exc)
    lowered = message.lower()
    if "api key not valid" in lowered or "api_key_invalid" in lowered:
        return ProviderError(
            "Google rejected the request: the GOOGLE_API_KEY is invalid. "
            "Set a valid key in .env and restart the server."
        )
    if "permission" in lowered or "unauthenticated" in lowered or "401" in lowered:
        return ProviderError(f"Google authentication failed: {message}")
    return ProviderError(f"Gemini request failed: {message}")


def _build_prompt(question: str, context: str) -> str:
    return f"""
You are Talking YouTube, a precise transcript-grounded assistant.

Rules:
- Answer only from the transcript context below.
- If the context is insufficient, say you do not know from these videos.
- Prefer concise answers with concrete details.
- Mention video ids or timestamps when they help the user verify the answer.

Transcript context:
{context}

Question:
{question}
""".strip()


def _format_context(matches: list[ChunkRecord]) -> str:
    blocks: list[str] = []
    for index, match in enumerate(matches, start=1):
        timestamp = match.timestamp or "00:00"
        blocks.append(
            f"[source {index} | video {match.video_id} | {timestamp}]\n{match.content}"
        )
    return "\n\n".join(blocks)


def _source_from_match(match: ChunkRecord) -> SourceChunk:
    return SourceChunk(
        video_id=match.video_id,
        source_url=match.source_url or "",
        timestamp=match.timestamp,
        text=_compact(match.content),
        score=match.score,
    )


def _compact(text: str, limit: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}..."


def _first_timestamp(text: str) -> str | None:
    match = TIMESTAMP_RE.search(text)
    return match.group("timestamp") if match else None


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _timestamp_to_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds
