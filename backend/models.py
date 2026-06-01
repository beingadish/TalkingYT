from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VideoInput(BaseModel):
    videos: list[str] = Field(..., min_length=1, max_length=12)
    languages: list[str] = Field(default_factory=lambda: ["en"])
    title: str | None = Field(default=None, max_length=120)

    @field_validator("videos")
    @classmethod
    def clean_videos(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        if not cleaned:
            raise ValueError("At least one YouTube URL or video id is required.")
        return cleaned

    @field_validator("languages")
    @classmethod
    def clean_languages(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned or ["en"]


class VideoSummary(BaseModel):
    video_id: str
    source_url: str
    transcript_characters: int
    snippet_count: int


class SessionSummary(BaseModel):
    session_id: str
    title: str | None = None
    created_at: datetime
    videos: list[VideoSummary]
    total_chunks: int
    total_transcript_characters: int


class IndexResponse(SessionSummary):
    status: Literal["indexed"]
    message: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=8)
    message: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=12)
    evaluate: bool = True


class AddVideosRequest(BaseModel):
    videos: list[str] = Field(..., min_length=1, max_length=12)
    languages: list[str] = Field(default_factory=lambda: ["en"])

    @field_validator("videos")
    @classmethod
    def clean_videos(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped and stripped not in cleaned:
                cleaned.append(stripped)
        if not cleaned:
            raise ValueError("At least one YouTube URL or video id is required.")
        return cleaned

    @field_validator("languages")
    @classmethod
    def clean_languages(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned or ["en"]


class SourceChunk(BaseModel):
    video_id: str
    source_url: str
    timestamp: str | None = None
    text: str
    score: float | None = None


class RagasEvaluation(BaseModel):
    metric: str = "answer_relevancy"
    status: Literal["scored", "skipped", "failed", "unavailable"]
    score: float | None = None
    reason: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk]
    evaluation: RagasEvaluation


class MessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: list[SourceChunk] = Field(default_factory=list)
    evaluation: RagasEvaluation | None = None
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    api: str
    has_google_api_key: bool
    active_sessions: int

