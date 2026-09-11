import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


DEFAULT_CANDIDATE_K = 20
DEFAULT_RRF_K = 60
TURKISH_CASE_MAP = str.maketrans(
    {
        "I": "ı",
        "İ": "i",
    }
)


@dataclass
class HybridSearchResult:
    payload: dict
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


# BM25 için metni küçük harfli kelimelere ayırır.
def tokenize_text(text: str) -> list[str]:
    normalized_text = (
        str(text)
        .translate(TURKISH_CASE_MAP)
        .casefold()
    )

    return re.findall(
        r"\w+",
        normalized_text,
        flags=re.UNICODE,
    )


# Chunk metinlerinden BM25 indeksi oluşturur.
def build_bm25_index(
    chunks: list[dict],
) -> BM25Okapi:
    if not chunks:
        raise ValueError(
            "BM25 indeksi için chunk bulunamadı."
        )

    tokenized_corpus = [
        tokenize_text(chunk["text"])
        for chunk in chunks
    ]

    if not any(tokenized_corpus):
        raise ValueError(
            "BM25 indeksi için kullanılabilir metin bulunamadı."
        )

    return BM25Okapi(tokenized_corpus)


# Sorguyla kelime bakımından eşleşen chunk'ları bulur.
def search_bm25(
    query: str,
    chunks: list[dict],
    bm25_index: BM25Okapi,
    top_k: int = DEFAULT_CANDIDATE_K,
) -> list[HybridSearchResult]:
    if top_k <= 0:
        raise ValueError(
            "BM25 Top-K sıfırdan büyük olmalıdır."
        )

    query_tokens = tokenize_text(query)

    if not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (
            -float(scores[index]),
            index,
        ),
    )

    results = []

    for chunk_index in ranked_indices:
        score = float(scores[chunk_index])

        document_terms = bm25_index.doc_freqs[
            chunk_index
        ]

        # Sorguyla ortak kelimesi bulunmayan chunk'ları alma.
        has_matching_term = any(
            token in document_terms
            for token in query_tokens
        )

        if not has_matching_term:
            continue

        chunk = chunks[chunk_index]
        payload = chunk["metadata"].copy()
        payload["text"] = chunk["text"]

        results.append(
            HybridSearchResult(
                payload=payload,
                score=score,
                lexical_rank=len(results) + 1,
            )
        )

        if len(results) >= top_k:
            break

    return results


# Dense ve BM25 sıralamalarını RRF ile birleştirir.
def reciprocal_rank_fusion(
    dense_results: list,
    lexical_results: list[HybridSearchResult],
    top_k: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[HybridSearchResult]:
    if top_k <= 0:
        raise ValueError(
            "Hybrid Top-K sıfırdan büyük olmalıdır."
        )

    if rrf_k <= 0:
        raise ValueError(
            "RRF sabiti sıfırdan büyük olmalıdır."
        )

    fused_results: dict[str, dict] = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        payload = result.payload or {}
        chunk_id = payload.get("chunk_id")

        if not chunk_id:
            continue

        fused_results[chunk_id] = {
            "payload": payload.copy(),
            "score": 1 / (rrf_k + rank),
            "dense_rank": rank,
            "lexical_rank": None,
        }

    for rank, result in enumerate(
        lexical_results,
        start=1,
    ):
        payload = result.payload or {}
        chunk_id = payload.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                "payload": payload.copy(),
                "score": 0.0,
                "dense_rank": None,
                "lexical_rank": rank,
            }
        else:
            fused_results[chunk_id][
                "lexical_rank"
            ] = rank

        fused_results[chunk_id]["score"] += (
            1 / (rrf_k + rank)
        )

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: (
            -item["score"],
            item["dense_rank"]
            if item["dense_rank"] is not None
            else float("inf"),
            item["lexical_rank"]
            if item["lexical_rank"] is not None
            else float("inf"),
        ),
    )

    return [
        HybridSearchResult(
            payload=item["payload"],
            score=item["score"],
            dense_rank=item["dense_rank"],
            lexical_rank=item["lexical_rank"],
        )
        for item in ranked_results[:top_k]
    ]
