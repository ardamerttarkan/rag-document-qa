from time import perf_counter

from embeddings import load_embedding_model
from llm import generate_answer
from retrieval import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    embed_query,
    search_similar_chunks,
)
from vector_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)


NO_RESULT_ANSWER = (
    "Dokümanlarda soruyla yeterince ilgili "
    "bir bilgi bulunamadı."
)


def build_context(results: list) -> str:
    context_parts = []

    for source_number, result in enumerate(results, start=1):
        payload = result.payload or {}
        text = (payload.get("text") or "").strip()

        if not text:
            continue

        document_name = (
            payload.get("document_name")
            or "Bilinmeyen doküman"
        )

        page = payload.get("page")
        page_text = (
            str(page)
            if page is not None
            else "Bilinmiyor"
        )

        context_part = (
            f"[Kaynak {source_number}]\n"
            f"Doküman: {document_name}\n"
            f"Sayfa: {page_text}\n"
            f"Metin: {text}"
        )

        context_parts.append(context_part)

    return "\n\n".join(context_parts)


def build_sources(results: list) -> list[dict]:
    sources = []

    for source_number, result in enumerate(results, start=1):
        payload = result.payload or {}

        source = {
            "source_number": source_number,
            "document_name": payload.get("document_name"),
            "page": payload.get("page"),
            "chunk_index": payload.get("chunk_index"),
            "chunk_id": payload.get("chunk_id"),
            "score": round(float(result.score), 4),
            "text": payload.get("text"),
        }

        sources.append(source)

    return sources


def answer_question(
    question: str,
    embedding_model,
    qdrant_client,
) -> dict:
    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Kullanıcı sorusu boş olamaz.")

    retrieval_start = perf_counter()

    query_vector = embed_query(
        query=cleaned_question,
        model=embedding_model,
    )

    results = search_similar_chunks(
        client=qdrant_client,
        query_vector=query_vector,
        top_k=DEFAULT_TOP_K,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
    )

    retrieval_ms = (
        perf_counter() - retrieval_start
    ) * 1000

    if not results:
        return {
            "question": cleaned_question,
            "answer": NO_RESULT_ANSWER,
            "context": "",
            "sources": [],
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_seconds": 0.0,
        }

    context = build_context(results)

    generation_start = perf_counter()

    answer = generate_answer(
        question=cleaned_question,
        context=context,
    )

    generation_seconds = (
        perf_counter() - generation_start
    )

    return {
        "question": cleaned_question,
        "answer": answer,
        "context": context,
        "sources": build_sources(results),
        "retrieval_ms": round(retrieval_ms, 2),
        "generation_seconds": round(
            generation_seconds,
            2,
        ),
    }


if __name__ == "__main__":
    question = input("Sorunuzu yazın: ").strip()

    embedding_model, _ = load_embedding_model()
    qdrant_client = create_qdrant_client()

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: "
            f"{COLLECTION_NAME}"
        )

    result = answer_question(
        question=question,
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
    )

    if result["context"]:
        print("\nOluşturulan context:")
        print("-" * 50)
        print(result["context"])

    print("\nModel cevabı:")
    print("-" * 50)
    print(result["answer"])