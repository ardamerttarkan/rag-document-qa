import json
from pathlib import Path
from time import perf_counter

from embeddings import load_embedding_model
from retrieval import embed_query, search_similar_chunks
from vector_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

TOP_K = 3
SCORE_THRESHOLD = 0.80


# Değerlendirme sorularını JSON dosyasından yükler.
def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "Değerlendirme dosyası bir JSON listesi olmalıdır."
        )

    return questions


# Beklenen sonucun arama sonuçlarındaki sırasını bulur.
def find_expected_result_rank(
    results: list,
    expected_document: str,
    expected_pages: list[int],
    expected_keywords: list[str],
) -> int | None:
    normalized_keywords = [
        keyword.casefold()
        for keyword in expected_keywords
    ]

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        document_name = payload.get("document_name")
        page = payload.get("page")
        text = (payload.get("text") or "").casefold()

        document_matches = (
            document_name == expected_document
        )

        page_matches = (
            not expected_pages
            or page in expected_pages
        )

        keywords_match = all(
            keyword in text
            for keyword in normalized_keywords
        )

        if (
            document_matches
            and page_matches
            and keywords_match
        ):
            return rank

    return None


# Arama sonuçlarını test senaryosuna göre değerlendirir.
def evaluate_results(
    test_case: dict,
    results: list,
) -> tuple[bool | None, str]:
    category = test_case["category"]

    if category == "answerable":
        expected_rank = find_expected_result_rank(
            results=results,
            expected_document=test_case[
                "expected_document"
            ],
            expected_pages=test_case[
                "expected_pages"
            ],
            expected_keywords=test_case[
                "expected_keywords"
            ],
        )

        if expected_rank is not None:
            return (
                True,
                f"Beklenen içerik "
                f"{expected_rank}. sırada bulundu",
            )

        return (
            False,
            "Beklenen içerik Top-K içinde bulunamadı",
        )

    if category == "unrelated":
        if not results:
            return (
                True,
                "Alakasız soru doğru şekilde elendi",
            )

        return (
            False,
            "Alakasız soru sonuç döndürdü",
        )

    if category == "related_unanswerable":
        return (
            None,
            "Generation aşamasında değerlendirilecek",
        )

    raise ValueError(
        f"Bilinmeyen soru kategorisi: {category}"
    )


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

    evaluated_count = 0
    passed_count = 0
    false_negative_count = 0
    false_positive_count = 0
    total_duration = 0.0

    print("Top-K:", TOP_K)
    print("Score threshold:", SCORE_THRESHOLD)
    print("Toplam soru:", len(questions))
    print("-" * 70)

    for test_case in questions:
        question = test_case["question"]

        start_time = perf_counter()

        query_vector = embed_query(
            query=question,
            model=embedding_model,
        )

        results = search_similar_chunks(
            client=qdrant_client,
            query_vector=query_vector,
            top_k=TOP_K,
            score_threshold=SCORE_THRESHOLD,
        )

        duration = perf_counter() - start_time
        total_duration += duration

        passed, explanation = evaluate_results(
            test_case=test_case,
            results=results,
        )

        if passed is True:
            status = "BAŞARILI"
            evaluated_count += 1
            passed_count += 1

        elif passed is False:
            status = "BAŞARISIZ"
            evaluated_count += 1

            if test_case["category"] == "answerable":
                false_negative_count += 1

            if test_case["category"] == "unrelated":
                false_positive_count += 1

        else:
            status = "BİLGİ"

        top_score = (
            f"{results[0].score:.4f}"
            if results
            else "-"
        )

        print(
            f"{test_case['id']} | "
            f"{status} | "
            f"sonuç={len(results)} | "
            f"en yüksek skor={top_score} | "
            f"{duration * 1000:.1f} ms"
        )
        print("Soru:", question)
        print("Açıklama:", explanation)
        print("-" * 70)

    success_rate = (
        passed_count / evaluated_count * 100
        if evaluated_count
        else 0.0
    )

    average_duration = (
        total_duration / len(questions)
        if questions
        else 0.0
    )

    print("\nDEĞERLENDİRME ÖZETİ")
    print("Değerlendirilen soru:", evaluated_count)
    print("Başarılı:", passed_count)
    print("False negative:", false_negative_count)
    print("False positive:", false_positive_count)
    print(f"Başarı oranı: %{success_rate:.2f}")
    print(
        f"Ortalama retrieval süresi: "
        f"{average_duration * 1000:.1f} ms"
    )
