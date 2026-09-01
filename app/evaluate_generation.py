import json
import re
from pathlib import Path
from time import perf_counter

from embeddings import load_embedding_model
from evaluation import load_questions
from llm import generate_answer
from rag import build_context
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "generation_results.json"
)

NO_ANSWER_MESSAGE = (
    "Bu sorunun cevabı verilen "
    "dokümanlarda bulunamadı."
)

KEYWORD_COVERAGE_THRESHOLD = 0.50

SOURCE_PATTERN = re.compile(
    r"\[Kaynak\s+(\d+)\]"
)


def normalize_text(text: str) -> str:
    return " ".join(
        text.casefold().split()
    )


def calculate_keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> tuple[float, list[str]]:
    if not expected_keywords:
        return 1.0, []

    normalized_answer = normalize_text(answer)

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if normalize_text(keyword) in normalized_answer
    ]

    coverage = (
        len(matched_keywords)
        / len(expected_keywords)
    )

    return coverage, matched_keywords


def extract_source_numbers(
    answer: str,
) -> list[int]:
    matches = SOURCE_PATTERN.findall(answer)

    return [
        int(source_number)
        for source_number in matches
    ]


def is_refusal(answer: str) -> bool:
    return (
        normalize_text(answer)
        == normalize_text(NO_ANSWER_MESSAGE)
    )


def evaluate_generation_result(
    test_case: dict,
    results: list,
    answer: str,
) -> tuple[bool, str, float | None, list[int]]:
    should_answer = test_case["should_answer"]

    if should_answer:
        if not results:
            return (
                False,
                "Retrieval sonuç getirmedi",
                0.0,
                [],
            )

        if is_refusal(answer):
            return (
                False,
                "Model cevaplanabilir soruyu reddetti",
                0.0,
                [],
            )

        keyword_coverage, _ = (
            calculate_keyword_coverage(
                answer=answer,
                expected_keywords=test_case[
                    "expected_keywords"
                ],
            )
        )

        source_numbers = extract_source_numbers(answer)

        sources_valid = (
            bool(source_numbers)
            and all(
                1 <= source_number <= len(results)
                for source_number in source_numbers
            )
        )

        coverage_valid = (
            keyword_coverage
            >= KEYWORD_COVERAGE_THRESHOLD
        )

        if not coverage_valid:
            return (
                False,
                "Cevaptaki anahtar kelime oranı yetersiz",
                keyword_coverage,
                source_numbers,
            )

        if not sources_valid:
            return (
                False,
                "Geçerli kaynak referansı bulunamadı",
                keyword_coverage,
                source_numbers,
            )

        return (
            True,
            "Cevap, anahtar kelime ve kaynak kontrolü başarılı",
            keyword_coverage,
            source_numbers,
        )

    if not results:
        return (
            True,
            "Soru retrieval aşamasında güvenli şekilde elendi",
            None,
            [],
        )

    if is_refusal(answer):
        return (
            True,
            "Model dokümanda olmayan cevabı üretmeyi reddetti",
            None,
            [],
        )

    return (
        False,
        "Model dokümanda bulunmayan bir cevap üretti",
        None,
        extract_source_numbers(answer),
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

    report = []

    passed_count = 0
    llm_call_count = 0
    total_retrieval_duration = 0.0
    total_generation_duration = 0.0

    print("Top-K:", DEFAULT_TOP_K)
    print(
        "Score threshold:",
        DEFAULT_SCORE_THRESHOLD,
    )
    print("Toplam soru:", len(questions))
    print("-" * 70)

    for test_case in questions:
        question = test_case["question"]

        retrieval_start = perf_counter()

        query_vector = embed_query(
            query=question,
            model=embedding_model,
        )

        results = search_similar_chunks(
            client=qdrant_client,
            query_vector=query_vector,
            top_k=DEFAULT_TOP_K,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
        )

        retrieval_duration = (
            perf_counter() - retrieval_start
        )

        total_retrieval_duration += (
            retrieval_duration
        )

        answer = ""
        generation_duration = 0.0
        llm_called = False

        if results:
            context = build_context(results)

            generation_start = perf_counter()

            answer = generate_answer(
                question=question,
                context=context,
            )

            generation_duration = (
                perf_counter() - generation_start
            )

            total_generation_duration += (
                generation_duration
            )

            llm_call_count += 1
            llm_called = True

        (
            passed,
            explanation,
            keyword_coverage,
            source_numbers,
        ) = evaluate_generation_result(
            test_case=test_case,
            results=results,
            answer=answer,
        )

        if passed:
            passed_count += 1
            status = "BAŞARILI"
        else:
            status = "BAŞARISIZ"

        coverage_text = (
            "-"
            if keyword_coverage is None
            else f"%{keyword_coverage * 100:.0f}"
        )

        print(
            f"{test_case['id']} | "
            f"{status} | "
            f"sonuç={len(results)} | "
            f"LLM={'evet' if llm_called else 'hayır'} | "
            f"kelime={coverage_text}"
        )

        print("Soru:", question)
        print("Açıklama:", explanation)

        if source_numbers:
            print("Kaynak numaraları:", source_numbers)

        if answer:
            print("Model cevabı:")
            print(answer)

        print(
            f"Süre: retrieval="
            f"{retrieval_duration * 1000:.1f} ms, "
            f"generation="
            f"{generation_duration:.2f} sn"
        )
        print("-" * 70)

        report.append(
            {
                "id": test_case["id"],
                "category": test_case["category"],
                "question": question,
                "passed": passed,
                "explanation": explanation,
                "result_count": len(results),
                "llm_called": llm_called,
                "keyword_coverage": keyword_coverage,
                "source_numbers": source_numbers,
                "answer": answer,
                "retrieval_duration_ms": (
                    retrieval_duration * 1000
                ),
                "generation_duration_seconds": (
                    generation_duration
                ),
            }
        )

    success_rate = (
        passed_count / len(questions) * 100
    )

    average_retrieval_ms = (
        total_retrieval_duration
        / len(questions)
        * 1000
    )

    average_generation_seconds = (
        total_generation_duration
        / llm_call_count
        if llm_call_count
        else 0.0
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nGENERATION DEĞERLENDİRME ÖZETİ")
    print("Başarılı:", passed_count)
    print("Başarısız:", len(questions) - passed_count)
    print(f"Başarı oranı: %{success_rate:.2f}")
    print(
        f"LLM çağrısı: "
        f"{llm_call_count}/{len(questions)}"
    )
    print(
        f"Ortalama retrieval süresi: "
        f"{average_retrieval_ms:.1f} ms"
    )
    print(
        f"Ortalama generation süresi: "
        f"{average_generation_seconds:.2f} sn"
    )
    print("Rapor:", REPORT_PATH)