from dataclasses import dataclass, field
import os


def _csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


# Values shipped in .env.example that must never be treated as a real key.
_PLACEHOLDER_API_KEYS = {"", "your_google_api_key"}


def _api_key_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value in _PLACEHOLDER_API_KEYS:
        return ""
    return value


@dataclass(frozen=True)
class Settings:
    google_api_key: str = field(default_factory=lambda: _api_key_env("GOOGLE_API_KEY"))
    google_chat_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    )
    google_eval_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_EVAL_MODEL", "gemini-2.5-flash")
    )
    google_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_EMBEDDING_MODEL",
            "models/gemini-embedding-001",
        )
    )
    chunk_size: int = field(
        default_factory=lambda: _int_env("TRANSCRIPT_CHUNK_SIZE", 1100)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _int_env("TRANSCRIPT_CHUNK_OVERLAP", 180)
    )
    default_top_k: int = field(default_factory=lambda: _int_env("DEFAULT_TOP_K", 5))
    cors_origins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cors_origins:
            object.__setattr__(
                self,
                "cors_origins",
                _csv_env(
                    "CORS_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000",
                ),
            )


def get_settings() -> Settings:
    return Settings()
