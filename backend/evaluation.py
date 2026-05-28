from __future__ import annotations

from dataclasses import dataclass
import sys
import types

from backend.config import Settings
from backend.models import RagasEvaluation


@dataclass(frozen=True)
class EvaluationPayload:
    question: str
    answer: str
    contexts: list[str]


class RagasEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def answer_relevancy(self, payload: EvaluationPayload) -> RagasEvaluation:
        if not self.settings.google_api_key:
            return RagasEvaluation(
                status="skipped",
                reason="GOOGLE_API_KEY is not configured.",
            )

        try:
            return await self._score_with_collections_api(payload)
        except ImportError as exc:
            return RagasEvaluation(
                status="unavailable",
                reason=f"RAGAS or google-genai is not installed: {exc}",
            )
        except Exception as first_error:
            try:
                return await self._score_with_legacy_api(payload)
            except ImportError as exc:
                return RagasEvaluation(
                    status="unavailable",
                    reason=f"Legacy RAGAS API is not installed: {exc}",
                )
            except Exception as second_error:
                return RagasEvaluation(
                    status="failed",
                    reason=f"{first_error}; legacy fallback failed: {second_error}",
                )

    async def _score_with_collections_api(
        self,
        payload: EvaluationPayload,
    ) -> RagasEvaluation:
        _ensure_ragas_langchain_community_vertexai()
        from google import genai
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy

        client = genai.Client(api_key=self.settings.google_api_key)
        llm = llm_factory(
            self.settings.google_eval_model,
            provider="google",
            client=client,
        )
        embeddings = GoogleEmbeddings(
            client=client,
            model=self.settings.google_embedding_model,
        )
        scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
        result = await scorer.ascore(
            user_input=payload.question,
            response=payload.answer,
        )
        return RagasEvaluation(status="scored", score=_extract_score(result))

    async def _score_with_legacy_api(self, payload: EvaluationPayload) -> RagasEvaluation:
        _ensure_ragas_langchain_community_vertexai()
        from google import genai
        from ragas import SingleTurnSample
        from ragas.embeddings import GoogleEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics import ResponseRelevancy

        client = genai.Client(api_key=self.settings.google_api_key)
        llm = llm_factory(
            self.settings.google_eval_model,
            provider="google",
            client=client,
        )
        embeddings = GoogleEmbeddings(
            client=client,
            model=self.settings.google_embedding_model,
        )
        sample = SingleTurnSample(
            user_input=payload.question,
            response=payload.answer,
            retrieved_contexts=payload.contexts,
        )
        scorer = ResponseRelevancy(llm=llm, embeddings=embeddings)
        result = await scorer.single_turn_ascore(sample)
        return RagasEvaluation(status="scored", score=_extract_score(result))


def _extract_score(result: object) -> float:
    value = getattr(result, "value", result)
    if isinstance(value, dict):
        value = next(iter(value.values()))
    return max(-1.0, min(1.0, float(value)))


def _ensure_ragas_langchain_community_vertexai() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim only.
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim
