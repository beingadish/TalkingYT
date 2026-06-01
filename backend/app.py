from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.db import create_pool, init_schema
from backend.models import (
    AddVideosRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexResponse,
    MessageItem,
    SessionSummary,
    VideoInput,
)
from backend.rag import MissingConfigurationError, ProviderError, RagEngine
from backend.repository import Repository, SessionNotFoundError, VideoNotFoundError
from backend.transcripts import TranscriptError


load_dotenv()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_schema(settings)
    pool = await create_pool(settings)
    app.state.pool = pool
    app.state.engine = RagEngine(settings, Repository(pool))
    try:
        yield
    finally:
        await pool.close()


app = FastAPI(
    title="Talking YouTube API",
    version="0.2.0",
    description=(
        "Transcript ingestion, pgvector retrieval, Gemini chat, persistent sessions "
        "and chat history, and optional RAGAS answer relevancy scoring."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> RagEngine:
    return app.state.engine


@app.get("/", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    active_sessions = await get_engine().repository.count_sessions()
    return HealthResponse(
        status="ok",
        api="talking-youtube",
        has_google_api_key=bool(settings.google_api_key),
        active_sessions=active_sessions,
    )


@app.post("/api/sessions", response_model=IndexResponse)
async def create_session(payload: VideoInput) -> IndexResponse:
    try:
        return await get_engine().create_session(
            videos=payload.videos,
            languages=payload.languages,
            title=payload.title,
        )
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TranscriptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    return await get_engine().list_sessions()


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    try:
        return await get_engine().get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    try:
        await get_engine().delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/videos", response_model=IndexResponse)
async def add_videos(session_id: str, payload: AddVideosRequest) -> IndexResponse:
    try:
        return await get_engine().add_videos(
            session_id=session_id,
            videos=payload.videos,
            languages=payload.languages,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TranscriptError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}/videos/{video_id}", response_model=SessionSummary)
async def remove_video(session_id: str, video_id: str) -> SessionSummary:
    try:
        return await get_engine().remove_video(session_id, video_id)
    except (SessionNotFoundError, VideoNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageItem])
async def get_messages(session_id: str) -> list[MessageItem]:
    try:
        return await get_engine().get_messages(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        return await get_engine().chat(
            session_id=payload.session_id,
            question=payload.message,
            top_k=payload.top_k,
            evaluate=payload.evaluate,
        )
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
