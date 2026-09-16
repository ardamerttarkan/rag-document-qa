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
    / "reranker_metrics.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "reranker_report.md"
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
from reranker import (  # noqa: E402
    DEFAULT_RERANKER_MODEL,
    load_reranker_model,
    rerank_results,
)
from evaluate_hybrid import (  # noqa: E402
    CHUNK_SIZE,
    OVERLAP,
    build_summary,
    evaluate_results,
)
from evaluate_retrieval import (  # noqa: E402
    MAX_TOP_K,
    NO_ANSWER_SCORE_THRESHOLD,
    RANKING_SCORE_THRESHOLD,
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


COLLECTION_NAME = "rag_documents_day15_reranker"


# Yeniden sıralanmış sonuçları JSON ile uyumlu kayıtlara dönüştürür.
def serialize_reranked_results(results: list) -> list[dict]:
    serialized = []

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}
        serialized.append(
            {
                "rank": rank,
                "reranker_score": round(
                    float(result.score),
                    6,
                ),
                "original_rank": result.original_rank,
                "original_score": round(
                    float(result.original_score),
                    6,
                ),
                "document": payload.get("document_name"),
                "page": payload.get("page"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
            }
        )

    return serialized


# Hibrit ve reranker sıralarını karşılaştırarak değişimin yönünü belirler.
def compare_ranks(
    hybrid_rank: int | None,
    reranker_rank: int | None,
) -> str:
    hybrid_value = (
        hybrid_rank
        if hybrid_rank is not None
        else float("inf")
    )
    reranker_value = (
        reranker_rank
        if reranker_rank is not None
        else float("inf")
    )

    if reranker_value < hybrid_value:
        return "improved"

    if reranker_value > hybrid_value:
        return "degraded"

    return "unchanged"


# Reranker etkisini soru bazında inceleyen hata analizini oluşturur.
def build_error_analysis(records: list[dict]) -> dict:
    details = []
    counts = {
        "improved": 0,
        "degraded": 0,
        "unchanged": 0,
    }

    for record in records:
        if not record["answerable"]:
            continue

        hybrid_rank = record["hybrid"][
            "first_relevant_rank"
        ]
        reranker_rank = record["reranker"][
            "first_relevant_rank"
        ]
        outcome = compare_ranks(
            hybrid_rank=hybrid_rank,
            reranker_rank=reranker_rank,
        )
        counts[outcome] += 1

        if outcome != "unchanged":
            details.append(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "outcome": outcome,
                    "hybrid_rank": hybrid_rank,
                    "reranker_rank": reranker_rank,
                }
            )

    return {
        **counts,
        "changed_questions": details,
    }


# Reranker değerlendirmesini Markdown raporuna yazar.
def write_report(evaluation: dict) -> None:
    dense = evaluation["summary"]["dense"]
    hybrid = evaluation["summary"]["hybrid"]
    reranker = evaluation["summary"]["reranker"]
    error_analysis = evaluation["error_analysis"]

    changed_rows = []

    for item in error_analysis["changed_questions"]:
        changed_rows.append(
            "| "
            f"{item['id']} | "
            f"{item['outcome']} | "
            f"{item['hybrid_rank']} | "
            f"{item['reranker_rank']} |"
        )

    if not changed_rows:
        changed_rows.append(
            "| - | Değişiklik yok | - | - |"
        )

    report = f"""# Reranker Benchmark ve Hata Analizi

## Yapılandırma

- Chunk size: `{CHUNK_SIZE}`
- Overlap: `{OVERLAP}`
- Top-K: `{MAX_TOP_K}`
- Reranker: `{DEFAULT_RERANKER_MODEL}`
- Çalışma cihazı: `CPU`

## Benchmark Sonuçları

| Yöntem | Recall@1 | Recall@3 | Recall@5 | MRR | Ortalama retrieval |
|---|---:|---:|---:|---:|---:|
| Dense | {dense['recall_at_1']:.2%} | {dense['recall_at_3']:.2%} | {dense['recall_at_5']:.2%} | {dense['mrr']:.4f} | {dense['average_retrieval_ms']:.2f} ms |
| Hybrid | {hybrid['recall_at_1']:.2%} | {hybrid['recall_at_3']:.2%} | {hybrid['recall_at_5']:.2%} | {hybrid['mrr']:.4f} | {hybrid['average_retrieval_ms']:.2f} ms |
| Hybrid + Reranker | {reranker['recall_at_1']:.2%} | {reranker['recall_at_3']:.2%} | {reranker['recall_at_5']:.2%} | {reranker['mrr']:.4f} | {reranker['average_retrieval_ms']:.2f} ms |

Ortalama yalnızca reranker inference süresi: {evaluation['average_reranker_ms']:.2f} ms.

## Hata Analizi

- İyileşen soru: `{error_analysis['improved']}`
- Gerileyen soru: `{error_analysis['degraded']}`
- Değişmeyen soru: `{error_analysis['unchanged']}`

| Soru | Durum | Hybrid sıra | Reranker sıra |
|---|---|---:|---:|
{chr(10).join(changed_rows)}

## İlk Yorum

Reranker yalnızca hybrid retrieval tarafından getirilen ilk {MAX_TOP_K} sonucu yeniden sıralamıştır. Bu nedenle aday kümesinde bulunmayan doğru bir chunk'ı geri getiremez; esas etkisi Recall@1, Recall@3 ve MRR üzerinde görülür. Nihai kullanım kararı başarı artışı ile CPU gecikmesi birlikte değerlendirilerek verilmelidir.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")


# Dense, hibrit ve reranker ölçümlerinin özetini terminalde gösterir.
def print_summary(evaluation: dict) -> None:
    print("\n" + "=" * 88)
    print("DENSE, HYBRID VE RERANKER KARŞILAŞTIRMASI")
    print("=" * 88)

    for method, label in (
        ("dense", "Dense"),
        ("hybrid", "Hybrid"),
        ("reranker", "Hybrid + Reranker"),
    ):
        metrics = evaluation["summary"][method]
        print(
            f"{label:<19} | "
            f"R@1: {metrics['recall_at_1']:.2%} | "
            f"R@3: {metrics['recall_at_3']:.2%} | "
            f"R@5: {metrics['recall_at_5']:.2%} | "
            f"MRR: {metrics['mrr']:.4f} | "
            f"arama: {metrics['average_retrieval_ms']:.2f} ms"
        )

    analysis = evaluation["error_analysis"]
    print(
        "\nReranker etkisi | "
        f"iyileşen: {analysis['improved']} | "
        f"gerileyen: {analysis['degraded']} | "
        f"değişmeyen: {analysis['unchanged']}"
    )
    print(
        "Ortalama reranker inference: "
        f"{evaluation['average_reranker_ms']:.2f} ms"
    )
    print("Sonuç dosyası:", RESULTS_PATH)
    print("Rapor dosyası:", REPORT_PATH)


# Reranker modelinin retrieval sıralamasına etkisini değerlendirir.
def evaluate_reranker() -> dict:
    dataset, questions = load_questions()

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Deney dokümanı bulunamadı: {DOCUMENT_PATH}"
        )

    configuration = {
        "name": "day15_reranker_benchmark",
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "top_k": MAX_TOP_K,
        "reranker_model": DEFAULT_RERANKER_MODEL,
        "reranker_device": "cpu",
    }
    documents = load_document(DOCUMENT_PATH)
    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
    )
    bm25_index = build_bm25_index(chunks)

    embedding_model, _ = load_embedding_model()
    query_vectors = prepare_query_vectors(
        questions,
        embedding_model,
    )
    reranker_model, reranker_load_seconds = (
        load_reranker_model()
    )
    client = create_qdrant_client()

    create_experiment_collection(client, COLLECTION_NAME)

    try:
        indexing = index_configuration(
            client=client,
            collection_name=COLLECTION_NAME,
            documents=documents,
            model=embedding_model,
            configuration=configuration,
        )
        question_records = []
        reranker_latencies = []

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

            reranked_results, reranker_ms = rerank_results(
                query=item["question"],
                results=hybrid_results,
                model=reranker_model,
                top_k=MAX_TOP_K,
            )
            reranker_latencies.append(reranker_ms)
            reranker_total_ms = hybrid_ms + reranker_ms

            dense_evaluation = evaluate_results(
                results=dense_results,
                item=item,
            )
            hybrid_evaluation = evaluate_results(
                results=hybrid_results,
                item=item,
            )
            reranker_evaluation = evaluate_results(
                results=reranked_results,
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
                    "results": serialize_results(hybrid_results),
                }
            )
            reranker_evaluation.update(
                {
                    "retrieval_ms": round(
                        reranker_total_ms,
                        3,
                    ),
                    "reranker_ms": round(reranker_ms, 3),
                    "results": serialize_reranked_results(
                        reranked_results
                    ),
                }
            )

            question_records.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answerable": answerable,
                    "dense": dense_evaluation,
                    "hybrid": hybrid_evaluation,
                    "reranker": reranker_evaluation,
                }
            )

            print(
                f"[{index:02d}/{len(questions)}] "
                f"{item['id']} | "
                f"dense: {dense_evaluation.get('first_relevant_rank')} | "
                f"hybrid: {hybrid_evaluation.get('first_relevant_rank')} | "
                f"reranker: "
                f"{reranker_evaluation.get('first_relevant_rank')} | "
                f"{reranker_ms:.2f} ms"
            )

        evaluation = {
            "dataset_name": dataset.get("dataset_name"),
            "dataset_version": dataset.get("version"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "question_count": len(questions),
            "configuration": configuration,
            "indexing": indexing,
            "reranker_load_seconds": round(
                reranker_load_seconds,
                3,
            ),
            "average_reranker_ms": round(
                mean(reranker_latencies),
                3,
            ),
            "summary": {
                "dense": build_summary(
                    question_records,
                    "dense",
                ),
                "hybrid": build_summary(
                    question_records,
                    "hybrid",
                ),
                "reranker": build_summary(
                    question_records,
                    "reranker",
                ),
            },
            "error_analysis": build_error_analysis(
                question_records
            ),
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
        print_summary(evaluation)

        return evaluation
    finally:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.close()


if __name__ == "__main__":
    evaluate_reranker()
