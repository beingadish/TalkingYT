# Talking YouTube

Chat with one or more YouTube videos by indexing their transcripts into a PostgreSQL + pgvector store, then asking transcript-grounded questions through a FastAPI backend and a minimal Next.js frontend. Sessions, embeddings, and chat history are persisted, and individual videos can be removed without disturbing the rest of the index.

## What is implemented

- Single or multi-video transcript ingestion
- YouTube URL and raw video id parsing
- Transcript chunking with timestamp labels
- Gemini embeddings with PostgreSQL + pgvector similarity retrieval
- Persistent sessions, embeddings, and chat history (survive restarts)
- Per-video removal that deletes only that video's embeddings
- Gemini chat responses constrained to retrieved transcript context
- Optional RAGAS answer relevancy scoring
- FastAPI endpoints for health, sessions, videos, messages, and chat
- Next.js frontend with a black/white monospace console UI
- Fully containerized stack (db, backend, frontend) via Docker Compose

## Project layout

```text
TalkingYoutube/
├── backend/
│   ├── app.py           # FastAPI routes + lifespan (pool/schema wiring)
│   ├── config.py        # Environment settings
│   ├── db.py            # Async psycopg pool + schema init (pgvector)
│   ├── repository.py    # Async CRUD: sessions/videos/chunks/messages + KNN
│   ├── evaluation.py    # RAGAS answer relevancy integration
│   ├── models.py        # API schemas
│   ├── rag.py           # Indexing, retrieval, generation (persists via repo)
│   └── transcripts.py   # YouTube transcript fetching
├── frontend/
│   ├── app/             # Next.js App Router UI
│   └── Dockerfile
├── indexing/            # Earlier prototype helpers
├── assistant/           # Earlier prototype prompt helper
├── utils/               # Earlier prototype utilities
├── main.py              # FastAPI entry point
├── Dockerfile           # Backend image
├── docker-compose.yml   # db (pgvector) + backend + frontend
└── requirements.txt
```

## Run with Docker Compose (recommended)

The whole stack (PostgreSQL + pgvector, backend, frontend) runs from one command.
All secrets and ports are configured through `.env`.

```bash
cp .env.example .env
# set GOOGLE_API_KEY and POSTGRES_PASSWORD in .env
docker compose up -d --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- Postgres is published on `POSTGRES_PORT` (default `5433` to avoid clashing with a local Postgres).

Data persists in the `db` volume, so sessions, embeddings, and chat history survive restarts.

## Backend setup (without Docker)

Running the backend directly requires a reachable PostgreSQL instance with the
`pgvector` extension available (the app creates the extension and schema on startup).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` and `DATABASE_URL` in `.env`, then start the API:

```bash
uvicorn main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

If the API is not on `http://localhost:8000`, set:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API

Create an indexed session:

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"videos":["https://www.youtube.com/watch?v=VIDEO_ID"],"languages":["en"]}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","message":"What is the core idea?","top_k":5,"evaluate":true}'
```

Add more videos to an existing session:

```bash
curl -X POST http://localhost:8000/api/sessions/SESSION_ID/videos \
  -H "Content-Type: application/json" \
  -d '{"videos":["https://www.youtube.com/watch?v=ANOTHER_ID"],"languages":["en"]}'
```

Remove a single video (deletes only that video's embeddings):

```bash
curl -X DELETE http://localhost:8000/api/sessions/SESSION_ID/videos/VIDEO_ID
```

Fetch persisted chat history:

```bash
curl http://localhost:8000/api/sessions/SESSION_ID/messages
```

Delete a whole session:

```bash
curl -X DELETE http://localhost:8000/api/sessions/SESSION_ID
```

## Notes

- Sessions, embeddings, and chat history are persisted in PostgreSQL + pgvector and survive restarts.
- Removing a video deletes only that video's chunks (via an `ON DELETE CASCADE` foreign key), leaving the rest of the index unchanged.
- Embeddings use `gemini-embedding-001` (3072 dimensions); retrieval uses exact cosine KNN scoped per session.
- RAGAS scoring is optional per request because it makes extra LLM and embedding calls.
- If RAGAS or `google-genai` is not installed, the answer still returns and the evaluation field reports `unavailable`.
