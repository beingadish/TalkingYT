# Talking YouTube

Chat with one or more YouTube videos by indexing their transcripts into a FAISS vector store, then asking transcript-grounded questions through a FastAPI backend and a minimal Next.js frontend.

## What is implemented

- Single or multi-video transcript ingestion
- YouTube URL and raw video id parsing
- Transcript chunking with timestamp labels
- Gemini embeddings with FAISS similarity retrieval
- Gemini chat responses constrained to retrieved transcript context
- Optional RAGAS answer relevancy scoring
- FastAPI endpoints for health, sessions, and chat
- Next.js frontend with a black/white monospace console UI

## Project layout

```text
TalkingYoutube/
├── backend/
│   ├── app.py           # FastAPI routes
│   ├── config.py        # Environment settings
│   ├── evaluation.py    # RAGAS answer relevancy integration
│   ├── models.py        # API schemas
│   ├── rag.py           # Indexing, retrieval, generation
│   └── transcripts.py   # YouTube transcript fetching
├── frontend/
│   └── app/             # Next.js App Router UI
├── indexing/            # Earlier prototype helpers
├── assistant/           # Earlier prototype prompt helper
├── utils/               # Earlier prototype utilities
├── main.py              # FastAPI entry point
└── requirements.txt
```

## Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`, then start the API:

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

## Notes

- Sessions are currently in memory. Restarting the API clears indexed videos.
- RAGAS scoring is optional per request because it makes extra LLM and embedding calls.
- If RAGAS or `google-genai` is not installed, the answer still returns and the evaluation field reports `unavailable`.
