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


if __name__ == "__main__":
    question = input("Sorunuzu yazın: ").strip()

    embedding_model, _ = load_embedding_model()

    query_vector = embed_query(
        query=question,
        model=embedding_model,
    )

    qdrant_client = create_qdrant_client()

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: {COLLECTION_NAME}"
        )

    results = search_similar_chunks(
        client=qdrant_client,
        query_vector=query_vector,
        top_k=DEFAULT_TOP_K,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
    )

    if not results:
        print(
            "\nDokümanlarda soruyla yeterince ilgili "
            "bir bilgi bulunamadı."
        )
        raise SystemExit(0)

    context = build_context(results)

    print("\nOluşturulan context:")
    print("-" * 50)
    print(context)
    
    answer = generate_answer(
    question=question,
    context=context,
)
    print("\nModel cevabı:")
    print("-" * 50)
    print(answer)