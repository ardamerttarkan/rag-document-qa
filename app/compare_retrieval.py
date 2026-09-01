from time import perf_counter

from embeddings import load_embedding_model
from evaluation import (
    evaluate_results,
    load_questions,
)
from retrieval import (
    embed_query,
    search_similar_chunks,
)
from vector_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)


TOP_K_VALUES = [1, 3, 5]
SCORE_THRESHOLDS = [0.75, 0.80, 0.85]


if __name__ == "__main__":
    questions = load_questions()

    embedding_model, _ = load_embedding_model()
    qdrant_client = create_qdrant_client()

    if not qdrant_client.collection_exists(
        COLLECTION_NAME
    ):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: "
            f"{COLLECTION_NAME}"
        )

    print("Query embedding'leri hazırlanıyor...")

    query_vectors = {}

    for test_case in questions:
        query_vectors[test_case["id"]] = embed_query(
            query=test_case["question"],
            model=embedding_model,
        )

    print("Embedding hazırlığı tamamlandı.\n")

    header = (
        f"{'Top-K':<7}"
        f"{'Eşik':<8}"
        f"{'Başarı':<10}"
        f"{'FN':<6}"
        f"{'FP':<6}"
        f"{'İlgili ctx':<13}"
        f"{'Ort. ms':<10}"
    )

    print(header)
    print("-" * len(header))

    for top_k in TOP_K_VALUES:
        for score_threshold in SCORE_THRESHOLDS:
            evaluated_count = 0
            passed_count = 0
            false_negative_count = 0
            false_positive_count = 0
            related_context_count = 0

            start_time = perf_counter()

            for test_case in questions:
                query_vector = query_vectors[
                    test_case["id"]
                ]

                results = search_similar_chunks(
                    client=qdrant_client,
                    query_vector=query_vector,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )

                category = test_case["category"]

                if category == "related_unanswerable":
                    if results:
                        related_context_count += 1

                    continue

                passed, _ = evaluate_results(
                    test_case=test_case,
                    results=results,
                )

                evaluated_count += 1

                if passed:
                    passed_count += 1
                elif category == "answerable":
                    false_negative_count += 1
                elif category == "unrelated":
                    false_positive_count += 1

            duration = perf_counter() - start_time

            success_rate = (
                passed_count
                / evaluated_count
                * 100
            )

            average_duration_ms = (
                duration
                / len(questions)
                * 1000
            )

            print(
                f"{top_k:<7}"
                f"{score_threshold:<8.2f}"
                f"%{success_rate:<9.2f}"
                f"{false_negative_count:<6}"
                f"{false_positive_count:<6}"
                f"{related_context_count}/3"
                f"{'':<10}"
                f"{average_duration_ms:<10.1f}"
            )