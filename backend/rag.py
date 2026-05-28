from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from uuid import uuid4

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - compatibility with older LangChain installs.
    from langchain.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - compatibility with older LangChain installs.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.config import Settings
from backend.evaluation import EvaluationPayload, RagasEvaluator
from backend.models import (
    ChatResponse,
    IndexResponse,
    SessionSummary,
    SourceChunk,
    VideoSummary,
)
from backend.transcripts import TranscriptBundle, TranscriptError, fetch_transcript, source_url


TIMESTAMP_RE = re.compile(r"\[(?P<timestamp>\d{2}:\d{2}(?::\d{2})?)\]")


class MissingConfigurationError(RuntimeError):
    """Raised when the configured model provider cannot be used."""


@dataclass
class VideoSession:
    session_id: str
    created_at: datetime
    title: str | None
    videos: list[VideoSummary]
    vector_store: FAISS
    total_chunks: int
    total_transcript_characters: int

    def summary(self) -> SessionSummary:
        return SessionSummary(
            session_id=self.session_id,
            title=self.title,
            created_at=self.created_at,
            videos=self.videos,
            total_chunks=self.total_chunks,
            total_transcript_characters=self.total_transcript_characters,
        )


class RagEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sessions: dict[str, VideoSession] = {}
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
        bundles = await asyncio.gather(
            *(asyncio.to_thread(fetch_transcript, video, languages) for video in videos)
        )

        docs: list[Document] = []
        summaries: list[VideoSummary] = []
        total_chars = 0
        for bundle in bundles:
            video_docs = self._documents_from_transcript(bundle)
            docs.extend(video_docs)
            transcript_characters = len(bundle.transcript_text)
            total_chars += transcript_characters
            summaries.append(
                VideoSummary(
                    video_id=bundle.video_id,
                    source_url=bundle.source_url,
                    transcript_characters=transcript_characters,
                    snippet_count=len(bundle.snippets),
                )
            )

        if not docs:
            raise TranscriptError("No transcript chunks were created.")

        vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            docs,
            self._get_embeddings(),
        )
        session_id = uuid4().hex[:12]
        session = VideoSession(
            session_id=session_id,
            created_at=datetime.now(timezone.utc),
            title=title,
            videos=summaries,
            vector_store=vector_store,
            total_chunks=len(docs),
            total_transcript_characters=total_chars,
        )
        self.sessions[session_id] = session
        return IndexResponse(
            **session.summary().model_dump(),
            status="indexed",
            message=f"Indexed {len(summaries)} video(s) into {len(docs)} transcript chunks.",
        )

    async def chat(
        self,
        session_id: str,
        question: str,
        top_k: int | None = None,
        evaluate: bool = True,
    ) -> ChatResponse:
        self._ensure_google_key()
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} was not found.")

        retrieval_count = top_k or self.settings.default_top_k
        matches = await asyncio.to_thread(
            session.vector_store.similarity_search_with_score,
            question,
            retrieval_count,
        )
        documents = [doc for doc, _score in matches]
        context = _format_context(documents)
        prompt = _build_prompt(question=question, context=context)

        response = await self._invoke_llm(prompt)
        sources = [_source_from_match(doc, score) for doc, score in matches]
        evaluation = (
            await self._evaluator.answer_relevancy(
                EvaluationPayload(
                    question=question,
                    answer=response,
                    contexts=[doc.page_content for doc in documents],
                )
            )
            if evaluate
            else await self._skipped_evaluation()
        )

        return ChatResponse(
            session_id=session_id,
            answer=response,
            sources=sources,
            evaluation=evaluation,
        )

    def list_sessions(self) -> list[SessionSummary]:
        return [session.summary() for session in self.sessions.values()]

    def get_session(self, session_id: str) -> SessionSummary:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} was not found.")
        return session.summary()

    def delete_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise KeyError(f"Session {session_id} was not found.")

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
        content = getattr(result, "content", result)
        return str(content).strip()

    async def _skipped_evaluation(self):
        from backend.models import RagasEvaluation

        return RagasEvaluation(status="skipped", reason="Evaluation was disabled.")


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


def _format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for index, doc in enumerate(documents, start=1):
        video_id = doc.metadata.get("video_id", "unknown")
        timestamp = doc.metadata.get("timestamp") or "00:00"
        blocks.append(
            f"[source {index} | video {video_id} | {timestamp}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def _source_from_match(doc: Document, score: float) -> SourceChunk:
    return SourceChunk(
        video_id=str(doc.metadata.get("video_id", "")),
        source_url=str(doc.metadata.get("source_url", "")),
        timestamp=doc.metadata.get("timestamp"),
        text=_compact(doc.page_content),
        score=float(score),
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

