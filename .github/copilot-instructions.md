# Copilot instructions for Talking YouTube

Chat with YouTube videos: transcripts are fetched, chunked, embedded into an in-memory
FAISS store, and answered with Gemini constrained to retrieved context. A FastAPI backend
serves a minimal Next.js console UI.

## Commands

Backend (run from repo root, inside the venv):

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000   # or: python main.py
curl http://localhost:8000/api/health
```

Frontend (from `frontend/`):

```bash
npm install
npm run dev        # http://localhost:3000
npm run build
npm run typecheck  # tsc --noEmit — the only static check in the repo
```

There is no automated test suite or Python linter configured. Verify backend changes by
hitting the endpoints (see `Readme.md` for `curl` examples against `/api/sessions` and
`/api/chat`). Verify frontend changes with `npm run typecheck`.

## Architecture

Request flow lives entirely in `backend/`:

- `main.py` re-exports `app` from `backend/app.py`; uvicorn targets `main:app`.
- `backend/app.py` — FastAPI routes only. Instantiates a single module-level `RagEngine`
  and translates engine exceptions into HTTP errors (`MissingConfigurationError` → 503,
  `TranscriptError` → 422, `KeyError` → 404).
- `backend/rag.py` — the core. `RagEngine` holds all `VideoSession`s in an in-memory dict
  (`self.sessions`), so **restarting the server clears every indexed video**. It fetches
  transcripts, chunks them, builds the FAISS store, runs similarity search, builds the
  prompt, and calls Gemini. Blocking work (transcript fetch, FAISS build, similarity
  search, sync LLM calls) is offloaded with `asyncio.to_thread`.
- `backend/transcripts.py` — `normalize_video_id` parses raw ids and watch/`youtu.be`/
  `embed`/`shorts`/`live` URLs; `fetch_transcript` returns a `TranscriptBundle`.
- `backend/evaluation.py` — optional RAGAS answer-relevancy scoring, isolated so failures
  never break a chat response (see conventions).
- `backend/models.py` — all Pydantic request/response schemas; `IndexResponse` extends
  `SessionSummary`.
- `backend/config.py` — `Settings` dataclass read from env vars via `get_settings()`.

Frontend is a single client component, `frontend/app/components/TalkingYoutubeConsole.tsx`,
that calls the backend directly. The API base comes from `NEXT_PUBLIC_API_URL` (default
`http://localhost:8000`).

`indexing/`, `assistant/`, and `utils/` are earlier standalone prototype helpers. They are
**not imported by the backend** — do not wire new work through them; extend `backend/`.

## Conventions

- **Config only through `Settings`.** Add a field in `backend/config.py` (with an env-var
  `default_factory`) and document it in `.env.example`; never read `os.getenv` elsewhere.
- **Engine raises domain exceptions, routes map them to HTTP.** Keep `backend/app.py` thin;
  put logic and custom exceptions in the engine/modules.
- **Evaluation must degrade gracefully.** RAGAS/`google-genai` are optional; scoring
  returns a `RagasEvaluation` with status `scored`/`skipped`/`failed`/`unavailable` instead
  of raising. Preserve this so a chat answer always returns.
- **Transcript chunks carry timestamp metadata.** Chunks are built from `[mm:ss]`-prefixed
  lines; `video_id`, `timestamp`, and a deep-link `source_url` (`...&t=<seconds>s`) are
  attached so answers can cite verifiable timestamps. Keep this metadata when changing
  chunking or retrieval.
- **Optional/runtime-version imports use try/except fallbacks** (FAISS, text splitters,
  youtube-transcript-api error classes). Follow that pattern for new optional deps.
- Backend uses `from __future__ import annotations` and `str | None`-style typing; match it.
- Frontend is the Next.js App Router with React 19; UI logic stays in the single client
  component and talks to the backend over `fetch`.
