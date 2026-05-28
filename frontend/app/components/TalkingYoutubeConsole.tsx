"use client";

import {
  Activity,
  Bot,
  CircleAlert,
  LoaderCircle,
  Play,
  Send,
  Trash2,
  Youtube
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";

type VideoSummary = {
  video_id: string;
  source_url: string;
  transcript_characters: number;
  snippet_count: number;
};

type SessionSummary = {
  session_id: string;
  title?: string | null;
  created_at: string;
  videos: VideoSummary[];
  total_chunks: number;
  total_transcript_characters: number;
};

type SourceChunk = {
  video_id: string;
  source_url: string;
  timestamp?: string | null;
  text: string;
  score?: number | null;
};

type RagasEvaluation = {
  metric: string;
  status: "scored" | "skipped" | "failed" | "unavailable";
  score?: number | null;
  reason?: string | null;
};

type ChatResponse = {
  session_id: string;
  answer: string;
  sources: SourceChunk[];
  evaluation: RagasEvaluation;
};

type HealthResponse = {
  status: "ok";
  has_google_api_key: boolean;
  active_sessions: number;
};

type Message =
  | {
      id: string;
      role: "user";
      content: string;
    }
  | {
      id: string;
      role: "assistant";
      content: string;
      sources: SourceChunk[];
      evaluation: RagasEvaluation;
    }
  | {
      id: string;
      role: "system";
      content: string;
    };

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

const EMPTY_EVALUATION: RagasEvaluation = {
  metric: "answer_relevancy",
  status: "skipped",
  reason: "No answer yet."
};

function uid() {
  return crypto.randomUUID();
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: 0 }).format(value);
}

function splitVideoInput(input: string) {
  return input
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function scoreLabel(evaluation: RagasEvaluation) {
  if (evaluation.status !== "scored" || evaluation.score == null) {
    return evaluation.status;
  }
  return `${Math.round(evaluation.score * 100)}%`;
}

export function TalkingYoutubeConsole() {
  const [videoInput, setVideoInput] = useState("");
  const [question, setQuestion] = useState("");
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: uid(),
      role: "system",
      content: "idle // awaiting tape"
    }
  ]);
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [evaluate, setEvaluate] = useState(true);
  const [topK, setTopK] = useState(5);
  const [error, setError] = useState<string | null>(null);

  const videos = useMemo(() => splitVideoInput(videoInput), [videoInput]);
  const busy = isIndexing || isAsking;
  const statusVerb = isIndexing ? "braiding" : isAsking ? "ruminating" : "listening";

  useEffect(() => {
    let ignore = false;

    async function ping() {
      try {
        const response = await fetch(`${API_URL}/api/health`);
        if (!response.ok) {
          throw new Error(`health ${response.status}`);
        }
        const data = (await response.json()) as HealthResponse;
        if (!ignore) {
          setApiHealth(data);
        }
      } catch {
        if (!ignore) {
          setApiHealth(null);
        }
      }
    }

    void ping();
    const id = window.setInterval(ping, 15000);
    return () => {
      ignore = true;
      window.clearInterval(id);
    };
  }, []);

  async function ingestVideos(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);

    if (!videos.length) {
      setError("drop at least one YouTube URL or video id");
      return;
    }

    setIsIndexing(true);
    setMessages((current) => [
      ...current,
      {
        id: uid(),
        role: "system",
        content: `braiding // ${videos.length} video${videos.length === 1 ? "" : "s"}`
      }
    ]);

    try {
      const response = await fetch(`${API_URL}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ videos, languages: ["en"] })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "indexing failed");
      }
      setSession(data as SessionSummary);
      setMessages((current) => [
        ...current,
        {
          id: uid(),
          role: "system",
          content: `distilled // ${(data as SessionSummary).total_chunks} chunks ready`
        }
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "indexing failed");
    } finally {
      setIsIndexing(false);
    }
  }

  async function ask(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || !session) {
      return;
    }

    setQuestion("");
    setError(null);
    setIsAsking(true);
    setMessages((current) => [
      ...current,
      {
        id: uid(),
        role: "user",
        content: trimmed
      }
    ]);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          message: trimmed,
          top_k: topK,
          evaluate
        })
      });
      const data = (await response.json()) as ChatResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in data ? data.detail : "chat failed");
      }
      const chat = data as ChatResponse;
      setMessages((current) => [
        ...current,
        {
          id: uid(),
          role: "assistant",
          content: chat.answer,
          sources: chat.sources,
          evaluation: chat.evaluation
        }
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "chat failed");
    } finally {
      setIsAsking(false);
    }
  }

  async function clearSession() {
    if (session) {
      await fetch(`${API_URL}/api/sessions/${session.session_id}`, { method: "DELETE" }).catch(
        () => undefined
      );
    }
    setSession(null);
    setMessages([
      {
        id: uid(),
        role: "system",
        content: "idle // awaiting tape"
      }
    ]);
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Bot size={18} aria-hidden="true" />
          </span>
          <div>
            <p className="eyebrow">Talking YouTube</p>
            <h1>transcript console</h1>
          </div>
        </div>
        <div className="health-strip" data-live={apiHealth?.status === "ok"}>
          <Activity size={16} aria-hidden="true" />
          <span>{apiHealth ? "api linked" : "api quiet"}</span>
          <span>{apiHealth?.has_google_api_key ? "key set" : "key missing"}</span>
          <span>{statusVerb}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="intake">
          <form onSubmit={ingestVideos} className={busy ? "agent-panel active" : "agent-panel"}>
            <div className="panel-head">
              <div>
                <p className="eyebrow">intake</p>
                <h2>tapes</h2>
              </div>
              <Youtube size={22} aria-hidden="true" />
            </div>

            <textarea
              value={videoInput}
              onChange={(event) => setVideoInput(event.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              spellCheck={false}
              rows={8}
            />

            <div className="inline-controls">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={evaluate}
                  onChange={(event) => setEvaluate(event.target.checked)}
                />
                <span>ragas</span>
              </label>

              <label className="k-control">
                <span>top-k</span>
                <input
                  type="number"
                  min={1}
                  max={12}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
            </div>

            <button className="primary-action" type="submit" disabled={isIndexing}>
              {isIndexing ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
              <span>{isIndexing ? "braiding" : "ingest"}</span>
            </button>
          </form>

          <div className="session-panel">
            <div className="panel-head compact">
              <div>
                <p className="eyebrow">session</p>
                <h2>{session ? session.session_id : "none"}</h2>
              </div>
              <button
                className="icon-button"
                type="button"
                onClick={clearSession}
                aria-label="Clear session"
                title="Clear session"
              >
                <Trash2 size={17} aria-hidden="true" />
              </button>
            </div>

            {session ? (
              <div className="metrics-grid">
                <div>
                  <span>videos</span>
                  <strong>{session.videos.length}</strong>
                </div>
                <div>
                  <span>chunks</span>
                  <strong>{formatNumber(session.total_chunks)}</strong>
                </div>
                <div>
                  <span>chars</span>
                  <strong>{formatNumber(session.total_transcript_characters)}</strong>
                </div>
              </div>
            ) : (
              <p className="muted">no indexed transcript</p>
            )}

            {session?.videos.map((video) => (
              <a
                className="video-row"
                href={video.source_url}
                target="_blank"
                rel="noreferrer"
                key={video.video_id}
              >
                <span>{video.video_id}</span>
                <small>{formatNumber(video.snippet_count)} lines</small>
              </a>
            ))}
          </div>
        </aside>

        <section className={busy ? "chat active" : "chat"}>
          <div className="chat-head">
            <div>
              <p className="eyebrow">dialogue</p>
              <h2>{session ? "ask the tape" : "index first"}</h2>
            </div>
            {busy && (
              <div className="thinking-pill">
                <span />
                <span />
                <span />
                {statusVerb}
              </div>
            )}
          </div>

          <div className="messages" aria-live="polite">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isAsking && (
              <div className="message assistant pending">
                <div className="message-meta">assistant</div>
                <div className="thinking-wave" />
                <p>ruminating // finding transcript pressure points</p>
              </div>
            )}
          </div>

          {error && (
            <div className="error-line">
              <CircleAlert size={16} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={ask} className="composer">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={session ? "ask anything from the indexed videos" : "ingest videos first"}
              disabled={!session || isAsking}
            />
            <button type="submit" disabled={!session || !question.trim() || isAsking}>
              {isAsking ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
              <span>send</span>
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "system") {
    return <div className="system-line">{message.content}</div>;
  }

  if (message.role === "user") {
    return (
      <article className="message user">
        <div className="message-meta">you</div>
        <p>{message.content}</p>
      </article>
    );
  }

  return (
    <article className="message assistant">
      <div className="message-meta">
        <span>assistant</span>
        <span>{scoreLabel(message.evaluation)}</span>
      </div>
      <p className="answer-text">{message.content}</p>
      <div className="source-list">
        {message.sources.map((source, index) => (
          <a
            href={source.source_url}
            target="_blank"
            rel="noreferrer"
            className="source"
            key={`${source.video_id}-${index}`}
          >
            <span>
              {source.video_id}
              {source.timestamp ? ` @ ${source.timestamp}` : ""}
            </span>
            <small>{source.text}</small>
          </a>
        ))}
      </div>
      {message.evaluation.reason && (
        <p className="eval-note">{message.evaluation.reason}</p>
      )}
    </article>
  );
}

