import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"
RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "retrieval_metrics.json"
)

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from embeddings import load_embedding_model  # noqa: E402
from retrieval import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    embed_query,
    search_similar_chunks,
)
from vector_store import (  # noqa: E402
    COLLECTION_NAME,
    create_qdrant_client,
)


TOP_K_VALUES = (1, 3, 5)
MAX_TOP_K = max(TOP_K_VALUES)
RANKING_SCORE_THRESHOLD = -1.0
NO_ANSWER_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
TURKISH_CASE_MAP = str.maketrans({"I": "ı", "İ": "i"})


def load_questions() -> tuple[dict, list[dict]]:
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dosyası bulunamadı: {QUESTIONS_PATH}"
        )

    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    questions = dataset.get("questions")

    if not isinstance(questions, list) or not questions:
        raise ValueError(
            "questions.json içerisinde geçerli bir questions listesi yok."
        )

    declared_count = dataset.get("question_count")

    if declared_count is not None and declared_count != len(questions):
        raise ValueError(
            "question_count ile gerçek soru sayısı eşleşmiyor: "
            f"{declared_count} != {len(questions)}"
        )

    common_fields = {"id", "question", "question_type"}
    answerable_fields = {
        "expected_document",
        "expected_page",
        "expected_keywords",
    }

    for index, item in enumerate(questions, start=1):
        missing_fields = common_fields.difference(item)

        if item.get("answerable", True):
            missing_fields.update(answerable_fields.difference(item))

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"{index}. soruda eksik alanlar bulundu: {missing}"
            )

        if item.get("answerable", True):
            keywords = item.get("expected_keywords")

            if not isinstance(keywords, list) or not keywords:
                raise ValueError(
                    f"{item['id']} için expected_keywords boş olamaz."
                )

    return dataset, questions


def normalize_text(value: object) -> str:
    text = str(value).translate(TURKISH_CASE_MAP).casefold()
    return re.sub(r"\s+", " ", text).strip()


def matches_expected_content(payload: dict, item: dict) -> bool:
    actual_document = Path(
        str(payload.get("document_name", ""))
    ).name

    try:
        actual_page = int(payload.get("page"))
    except (TypeError, ValueError):
        return False

    if (
        actual_document != Path(item["expected_document"]).name
        or actual_page != int(item["expected_page"])
    ):
        return False

    chunk_text = normalize_text(payload.get("text", ""))

    return all(
        normalize_text(keyword) in chunk_text
        for keyword in item["expected_keywords"]
    )


def find_first_relevant_rank(results: list, item: dict) -> int | None:
    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        if matches_expected_content(payload=payload, item=item):
            return rank

    return None


def serialize_results(results: list) -> list[dict]:
    serialized = []

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        serialized.append(
            {
                "rank": rank,
                "score": round(float(result.score), 6),
                "document": payload.get("document_name"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
            }
        )

    return serialized


def calculate_average(values: list[float]) -> float:
    return round(mean(values), 3) if values else 0.0


def evaluate() -> dict:
    dataset, questions = load_questions()

    model_load_started = perf_counter()
    model, _ = load_embedding_model()
    model_load_ms = (perf_counter() - model_load_started) * 1000

    client = create_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: {COLLECTION_NAME}"
        )

    hit_counts = {top_k: 0 for top_k in TOP_K_VALUES}
    reciprocal_ranks = []
    total_latencies_ms = []
    embedding_latencies_ms = []
    search_latencies_ms = []
    answerable_count = 0
    no_answer_count = 0
    rejected_no_answer_count = 0
    question_results = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        answerable = item.get("answerable", True)
        score_threshold = (
            RANKING_SCORE_THRESHOLD
            if answerable
            else NO_ANSWER_SCORE_THRESHOLD
        )

        total_started = perf_counter()

        embedding_started = perf_counter()
        query_vector = embed_query(query=question, model=model)
        embedding_ms = (perf_counter() - embedding_started) * 1000

        search_started = perf_counter()
        results = search_similar_chunks(
            client=client,
            query_vector=query_vector,
            top_k=MAX_TOP_K,
            score_threshold=score_threshold,
        )
        search_ms = (perf_counter() - search_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000

        total_latencies_ms.append(total_ms)
        embedding_latencies_ms.append(embedding_ms)
        search_latencies_ms.append(search_ms)

        result_record = {
            "id": item["id"],
            "question": question,
            "question_type": item["question_type"],
            "answerable": answerable,
            "latency_ms": {
                "embedding": round(embedding_ms, 3),
                "search": round(search_ms, 3),
                "total": round(total_ms, 3),
            },
            "retrieved_results": serialize_results(results),
        }

        if answerable:
            answerable_count += 1
            first_relevant_rank = find_first_relevant_rank(
                results=results,
                item=item,
            )
            reciprocal_rank = (
                1 / first_relevant_rank
                if first_relevant_rank is not None
                else 0.0
            )
            reciprocal_ranks.append(reciprocal_rank)

            hits = {}

            for top_k in TOP_K_VALUES:
                hit = (
                    first_relevant_rank is not None
                    and first_relevant_rank <= top_k
                )
                hits[f"recall_at_{top_k}"] = int(hit)
                hit_counts[top_k] += int(hit)

            result_record.update(
                {
                    "expected_document": item["expected_document"],
                    "expected_page": item["expected_page"],
                    "expected_keywords": item["expected_keywords"],
                    "first_relevant_rank": first_relevant_rank,
                    "reciprocal_rank": round(reciprocal_rank, 6),
                    **hits,
                }
            )

            outcome_text = (
                f"doğru chunk sırası: {first_relevant_rank}"
                if first_relevant_rank is not None
                else "doğru chunk bulunamadı"
            )
        else:
            no_answer_count += 1
            rejected = not results
            rejected_no_answer_count += int(rejected)
            result_record.update(
                {
                    "no_answer_rejected": rejected,
                    "top_score": (
                        round(float(results[0].score), 6)
                        if results
                        else None
                    ),
                }
            )
            outcome_text = (
                "doğru biçimde reddedildi"
                if rejected
                else "ilgili görünen kaynak bulundu"
            )

        question_results.append(result_record)
        print(
            f"[{index:02d}/{len(questions)}] "
            f"{item['id']} | {outcome_text} | {total_ms:.2f} ms"
        )

    summary = {
        "answerable_question_count": answerable_count,
        "no_answer_question_count": no_answer_count,
        **{
            f"recall_at_{top_k}": round(
                hit_counts[top_k] / answerable_count,
                6,
            )
            for top_k in TOP_K_VALUES
        },
        "mrr": round(mean(reciprocal_ranks), 6),
        "no_answer_rejection_accuracy": (
            round(rejected_no_answer_count / no_answer_count, 6)
            if no_answer_count
            else None
        ),
        "average_latency_ms": calculate_average(total_latencies_ms),
        "average_embedding_ms": calculate_average(
            embedding_latencies_ms
        ),
        "average_search_ms": calculate_average(search_latencies_ms),
    }

    evaluation = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "configuration": {
            "collection_name": COLLECTION_NAME,
            "max_top_k": MAX_TOP_K,
            "ranking_score_threshold": RANKING_SCORE_THRESHOLD,
            "no_answer_score_threshold": NO_ANSWER_SCORE_THRESHOLD,
            "relevance_rule": (
                "document + page + all expected keywords in one chunk"
            ),
        },
        "model_load_ms": round(model_load_ms, 3),
        "summary": summary,
        "questions": question_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return evaluation


def print_summary(evaluation: dict) -> None:
    summary = evaluation["summary"]

    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION SONUÇLARI")
    print("=" * 50)
    print("Toplam soru:", evaluation["question_count"])
    print("Cevaplanabilir soru:", summary["answerable_question_count"])
    print("No-answer soru:", summary["no_answer_question_count"])
    print(f"Recall@1: {summary['recall_at_1']:.2%}")
    print(f"Recall@3: {summary['recall_at_3']:.2%}")
    print(f"Recall@5: {summary['recall_at_5']:.2%}")
    print(f"MRR: {summary['mrr']:.4f}")

    rejection_accuracy = summary["no_answer_rejection_accuracy"]

    if rejection_accuracy is not None:
        print(
            "No-answer reddetme başarısı: "
            f"{rejection_accuracy:.2%}"
        )

    print(
        "Ortalama toplam gecikme: "
        f"{summary['average_latency_ms']:.2f} ms"
    )
    print(
        "Ortalama embedding süresi: "
        f"{summary['average_embedding_ms']:.2f} ms"
    )
    print(
        "Ortalama Qdrant arama süresi: "
        f"{summary['average_search_ms']:.2f} ms"
    )
    print("Sonuç dosyası:", RESULTS_PATH)


if __name__ == "__main__":
    evaluation_result = evaluate()
    print_summary(evaluation_result)
