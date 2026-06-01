from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexResponse,
    SessionSummary,
    VideoInput,
)
from backend.rag import MissingConfigurationError, ProviderError, RagEngine
from backend.transcripts import TranscriptError


load_dotenv()

settings = get_settings()
engine = RagEngine(settings)

app = FastAPI(
    title="Talking YouTube API",
    version="0.1.0",
    description="Transcript ingestion, vector retrieval, Gemini chat, and optional RAGAS answer relevancy scoring.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        api="talking-youtube",
        has_google_api_key=bool(settings.google_api_key),
        active_sessions=len(engine.sessions),
    )


@app.post("/api/sessions", response_model=IndexResponse)
async def create_session(payload: VideoInput) -> IndexResponse:
    try:
        return await engine.create_session(
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
    return engine.list_sessions()


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
async def get_session(session_id: str) -> SessionSummary:
    try:
        return engine.get_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    try:
        engine.delete_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        return await engine.chat(
            session_id=payload.session_id,
            question=payload.message,
            top_k=payload.top_k,
            evaluate=payload.evaluate,
        )
    except MissingConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

