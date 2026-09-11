import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "hybrid_metrics.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "hybrid_report.md"
)
DOCUMENT_PATH = (
    PROJECT_ROOT
    / "pdf_samples"
    / "uyku_duzeni.pdf"
)

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chunking import chunk_documents  # noqa: E402
from embeddings import load_embedding_model  # noqa: E402
from hybrid_retrieval import (  # noqa: E402
    build_bm25_index,
    reciprocal_rank_fusion,
    search_bm25,
)
from ingest import load_document  # noqa: E402
from evaluate_retrieval import (  # noqa: E402
    MAX_TOP_K,
    NO_ANSWER_SCORE_THRESHOLD,
    RANKING_SCORE_THRESHOLD,
    TOP_K_VALUES,
    find_first_relevant_rank,
    load_questions,
    serialize_results,
)
from run_chunk_experiments import (  # noqa: E402
    create_experiment_collection,
    index_configuration,
    prepare_query_vectors,
    search_collection,
)
from vector_store import create_qdrant_client  # noqa: E402


COLLECTION_NAME = "rag_documents_day14_hybrid"
CHUNK_SIZE = 500
OVERLAP = 100


def evaluate_results(results: list, item: dict) -> dict:
    if item.get("answerable", True):
        first_relevant_rank = find_first_relevant_rank(
            results=results,
            item=item,
        )
        return {
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": (
                round(1 / first_relevant_rank, 6)
                if first_relevant_rank is not None
                else 0.0
            ),
        }

    return {
        "no_answer_rejected": not results,
        "top_score": (
            round(float(results[0].score), 6)
            if results
            else None
        ),
    }


def build_summary(records: list[dict], method: str) -> dict:
    answerable_records = [
        record
        for record in records
        if record["answerable"]
    ]
    no_answer_records = [
        record
        for record in records
        if not record["answerable"]
    ]

    summary = {
        "answerable_question_count": len(answerable_records),
        "no_answer_question_count": len(no_answer_records),
    }

    for top_k in TOP_K_VALUES:
        hits = sum(
            record[method]["first_relevant_rank"] is not None
            and record[method]["first_relevant_rank"] <= top_k
            for record in answerable_records
        )
        summary[f"recall_at_{top_k}"] = round(
            hits / len(answerable_records),
            6,
        )

    summary["mrr"] = round(
        mean(
            record[method]["reciprocal_rank"]
            for record in answerable_records
        ),
        6,
    )
    summary["no_answer_rejection_accuracy"] = (
        round(
            sum(
                record[method]["no_answer_rejected"]
                for record in no_answer_records
            )
            / len(no_answer_records),
            6,
        )
        if no_answer_records
        else None
    )
    summary["average_retrieval_ms"] = round(
        mean(
            record[method]["retrieval_ms"]
            for record in records
        ),
        3,
    )

    return summary


def write_report(evaluation: dict) -> None:
    dense = evaluation["summary"]["dense"]
    hybrid = evaluation["summary"]["hybrid"]

    report = f"""# Dense ve Hybrid Retrieval Karşılaştırması

## Yapılandırma

- Chunk size: `{CHUNK_SIZE}`
- Overlap: `{OVERLAP}`
- Top-K: `{MAX_TOP_K}`
- Hybrid yöntem: Dense + BM25 + Reciprocal Rank Fusion

## Sonuçlar

| Yöntem | Recall@1 | Recall@3 | Recall@5 | MRR | Ortalama retrieval |
|---|---:|---:|---:|---:|---:|
| Dense | {dense['recall_at_1']:.2%} | {dense['recall_at_3']:.2%} | {dense['recall_at_5']:.2%} | {dense['mrr']:.4f} | {dense['average_retrieval_ms']:.2f} ms |
| Hybrid | {hybrid['recall_at_1']:.2%} | {hybrid['recall_at_3']:.2%} | {hybrid['recall_at_5']:.2%} | {hybrid['mrr']:.4f} | {hybrid['average_retrieval_ms']:.2f} ms |

## İlk Yorum

Hybrid yöntemin Recall@3 değişimi: {(hybrid['recall_at_3'] - dense['recall_at_3']):+.2%}.

Hybrid yöntemin MRR değişimi: {(hybrid['mrr'] - dense['mrr']):+.4f}.

Hybrid retrieval, anlamsal dense arama ile kelime tabanlı BM25 sonuçlarını RRF yöntemiyle birleştirmiştir. Nihai yöntem seçilirken retrieval başarısı ile gecikme birlikte değerlendirilmelidir.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


def print_comparison(evaluation: dict) -> None:
    dense = evaluation["summary"]["dense"]
    hybrid = evaluation["summary"]["hybrid"]

    print("\n" + "=" * 74)
    print("DENSE VE HYBRID RETRIEVAL KARŞILAŞTIRMASI")
    print("=" * 74)

    for name, metrics in (
        ("Dense", dense),
        ("Hybrid", hybrid),
    ):
        print(
            f"{name:<7} | "
            f"R@1: {metrics['recall_at_1']:.2%} | "
            f"R@3: {metrics['recall_at_3']:.2%} | "
            f"R@5: {metrics['recall_at_5']:.2%} | "
            f"MRR: {metrics['mrr']:.4f} | "
            f"arama: {metrics['average_retrieval_ms']:.2f} ms"
        )

    print("Sonuç dosyası:", RESULTS_PATH)
    print("Rapor dosyası:", REPORT_PATH)


def evaluate_hybrid() -> dict:
    dataset, questions = load_questions()

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Deney dokümanı bulunamadı: {DOCUMENT_PATH}"
        )

    configuration = {
        "name": "day14_dense_vs_hybrid",
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k": MAX_TOP_K,
    }
    documents = load_document(DOCUMENT_PATH)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    bm25_index = build_bm25_index(chunks)
    model, _ = load_embedding_model()
    query_vectors = prepare_query_vectors(questions, model)
    client = create_qdrant_client()

    create_experiment_collection(client, COLLECTION_NAME)

    try:
        indexing = index_configuration(
            client=client,
            collection_name=COLLECTION_NAME,
            documents=documents,
            model=model,
            configuration=configuration,
        )
        question_records = []

        for index, item in enumerate(questions, start=1):
            answerable = item.get("answerable", True)
            threshold = (
                RANKING_SCORE_THRESHOLD
                if answerable
                else NO_ANSWER_SCORE_THRESHOLD
            )

            dense_results, dense_ms = search_collection(
                client=client,
                collection_name=COLLECTION_NAME,
                query_vector=query_vectors[item["id"]],
                score_threshold=threshold,
            )

            lexical_started = perf_counter()
            lexical_results = search_bm25(
                query=item["question"],
                chunks=chunks,
                bm25_index=bm25_index,
                top_k=MAX_TOP_K,
            )
            lexical_ms = (
                perf_counter() - lexical_started
            ) * 1000

            fusion_started = perf_counter()
            hybrid_results = reciprocal_rank_fusion(
                dense_results=dense_results,
                lexical_results=lexical_results,
                top_k=MAX_TOP_K,
            )
            fusion_ms = (
                perf_counter() - fusion_started
            ) * 1000
            hybrid_ms = dense_ms + lexical_ms + fusion_ms

            dense_evaluation = evaluate_results(
                results=dense_results,
                item=item,
            )
            hybrid_evaluation = evaluate_results(
                results=hybrid_results,
                item=item,
            )
            dense_evaluation.update(
                {
                    "retrieval_ms": round(dense_ms, 3),
                    "results": serialize_results(dense_results),
                }
            )
            hybrid_evaluation.update(
                {
                    "retrieval_ms": round(hybrid_ms, 3),
                    "lexical_ms": round(lexical_ms, 3),
                    "fusion_ms": round(fusion_ms, 3),
                    "results": serialize_results(hybrid_results),
                }
            )

            question_records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answerable": answerable,
                    "dense": dense_evaluation,
                    "hybrid": hybrid_evaluation,
                }
            )

            print(
                f"[{index:02d}/{len(questions)}] "
                f"{item['id']} | "
                f"dense: {dense_evaluation.get('first_relevant_rank')} | "
                f"hybrid: {hybrid_evaluation.get('first_relevant_rank')}"
            )

        evaluation = {
            "dataset_name": dataset.get("dataset_name"),
            "dataset_version": dataset.get("version"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "configuration": configuration,
            "indexing": indexing,
            "summary": {
                "dense": build_summary(
                    question_records,
                    "dense",
                ),
                "hybrid": build_summary(
                    question_records,
                    "hybrid",
                ),
            },
            "questions": question_records,
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(
            json.dumps(
                evaluation,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        write_report(evaluation)
        print_comparison(evaluation)

        return evaluation
    finally:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.close()


if __name__ == "__main__":
    evaluate_hybrid()
