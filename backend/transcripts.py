from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

try:
    from youtube_transcript_api._errors import VideoUnavailable
except ImportError:  # pragma: no cover - older youtube-transcript-api versions.
    class VideoUnavailable(Exception):
        pass


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(RuntimeError):
    """Raised when a video transcript cannot be fetched."""


@dataclass(frozen=True)
class TranscriptSnippet:
    text: str
    start: float
    duration: float


@dataclass(frozen=True)
class TranscriptBundle:
    video_id: str
    source_url: str
    snippets: list[TranscriptSnippet]

    @property
    def transcript_text(self) -> str:
        return " ".join(snippet.text for snippet in self.snippets)


def normalize_video_id(video: str) -> str:
    value = video.strip()
    if VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower().removeprefix("www.")
    path_parts = [part for part in parsed.path.split("/") if part]

    if host == "youtu.be" and path_parts:
        candidate = path_parts[0]
    elif host.endswith("youtube.com"):
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            candidate = query_id
        elif path_parts and path_parts[0] in {"embed", "shorts", "live"}:
            candidate = path_parts[1] if len(path_parts) > 1 else ""
        else:
            candidate = ""
    else:
        candidate = ""

    if not VIDEO_ID_RE.match(candidate):
        raise TranscriptError(f"Could not read a YouTube video id from: {video}")
    return candidate


def source_url(video_id: str, start_seconds: float | None = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    if start_seconds is None:
        return base
    return f"{base}&t={max(0, int(start_seconds))}s"


def fetch_transcript(video: str, languages: list[str]) -> TranscriptBundle:
    video_id = normalize_video_id(video)
    try:
        transcript_api = YouTubeTranscriptApi()
        try:
            raw_transcript = transcript_api.fetch(video_id=video_id, languages=languages)
        except TypeError:
            raw_transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=languages,
            )
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as exc:
        raise TranscriptError(f"No usable captions found for {video_id}: {exc}") from exc
    except Exception as exc:  # The library raises transport-specific exceptions too.
        raise TranscriptError(f"Transcript fetch failed for {video_id}: {exc}") from exc

    snippets = [_coerce_snippet(item) for item in raw_transcript]
    snippets = [snippet for snippet in snippets if snippet.text]
    if not snippets:
        raise TranscriptError(f"Transcript for {video_id} was empty.")

    return TranscriptBundle(
        video_id=video_id,
        source_url=source_url(video_id),
        snippets=snippets,
    )


def _coerce_snippet(item: object) -> TranscriptSnippet:
    if isinstance(item, dict):
        return TranscriptSnippet(
            text=str(item.get("text", "")).strip(),
            start=float(item.get("start", 0.0) or 0.0),
            duration=float(item.get("duration", 0.0) or 0.0),
        )

    return TranscriptSnippet(
        text=str(getattr(item, "text", "")).strip(),
        start=float(getattr(item, "start", 0.0) or 0.0),
        duration=float(getattr(item, "duration", 0.0) or 0.0),
    )
