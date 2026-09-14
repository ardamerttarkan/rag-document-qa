from dataclasses import dataclass
from time import perf_counter

from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/"
    "mmarco-mMiniLMv2-L12-H384-v1"
)
DEFAULT_RERANK_BATCH_SIZE = 8
DEFAULT_RERANK_TOP_K = 5
DEFAULT_RERANK_MAX_LENGTH = 512


@dataclass
class RerankedResult:
    payload: dict
    score: float
    original_rank: int
    original_score: float


# CPU üzerinde çalışacak reranker modelini yükler.
def load_reranker_model(
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> tuple[CrossEncoder, float]:
    started = perf_counter()

    model = CrossEncoder(
        model_name,
        device="cpu",
        max_length=DEFAULT_RERANK_MAX_LENGTH,
    )

    load_duration = perf_counter() - started
    return model, load_duration


# Retrieval sonuçlarını soru-chunk ilişkisine göre yeniden sıralar.
def rerank_results(
    query: str,
    results: list,
    model: CrossEncoder,
    top_k: int = DEFAULT_RERANK_TOP_K,
    batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
) -> tuple[list[RerankedResult], float]:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Reranker sorgusu boş olamaz.")

    if top_k <= 0:
        raise ValueError("Reranker Top-K sıfırdan büyük olmalıdır.")

    if batch_size <= 0:
        raise ValueError(
            "Reranker batch size sıfırdan büyük olmalıdır."
        )

    if not results:
        return [], 0.0

    pairs = []

    for result in results:
        payload = result.payload or {}
        text = str(payload.get("text", "")).strip()

        if not text:
            raise ValueError(
                "Reranker sonucunda chunk metni bulunamadı."
            )

        pairs.append((cleaned_query, text))

    started = perf_counter()

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    rerank_ms = (perf_counter() - started) * 1000
    reranked_results = []

    for original_rank, (result, score) in enumerate(
        zip(results, scores),
        start=1,
    ):
        score_value = (
            float(score.item())
            if hasattr(score, "item")
            else float(score)
        )

        reranked_results.append(
            RerankedResult(
                payload=(result.payload or {}).copy(),
                score=score_value,
                original_rank=original_rank,
                original_score=float(result.score),
            )
        )

    reranked_results.sort(
        key=lambda result: (
            -result.score,
            result.original_rank,
        )
    )

    return reranked_results[:top_k], rerank_ms
