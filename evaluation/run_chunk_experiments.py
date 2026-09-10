import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "chunk_experiments.json"
)
DOCUMENT_PATH = (
    PROJECT_ROOT
    / "pdf_samples"
    / "uyku_duzeni.pdf"
)

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chunking import chunk_documents  # noqa: E402
from embeddings import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    embed_chunks,
    load_embedding_model,
)
from ingest import load_document  # noqa: E402
from retrieval import embed_query  # noqa: E402
from vector_store import (  # noqa: E402
    EMBEDDING_DIMENSION,
    create_qdrant_client,
    embedded_chunk_to_point,
)

from evaluate_retrieval import (  # noqa: E402
    MAX_TOP_K,
    NO_ANSWER_SCORE_THRESHOLD,
    RANKING_SCORE_THRESHOLD,
    TOP_K_VALUES,
    find_first_relevant_rank,
    load_questions,
)


EXPERIMENTS = (
    {
        "name": "small_chunks",
        "chunk_size": 300,
        "overlap": 50,
    },
    {
        "name": "baseline",
        "chunk_size": 500,
        "overlap": 100,
    },
    {
        "name": "large_chunks",
        "chunk_size": 800,
        "overlap": 150,
    },
)


def build_collection_name(configuration: dict) -> str:
    return (
        "rag_documents_day13_"
        f"{configuration['chunk_size']}_"
        f"{configuration['overlap']}"
    )


def create_experiment_collection(client, collection_name: str) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )


def index_configuration(
    client,
    collection_name: str,
    documents: list[dict],
    model,
    configuration: dict,
) -> dict:
    chunking_started = perf_counter()
    chunks = chunk_documents(
        documents=documents,
        chunk_size=configuration["chunk_size"],
        overlap=configuration["overlap"],
    )
    chunking_ms = (perf_counter() - chunking_started) * 1000

    embedding_started = perf_counter()
    embedded_chunks, _ = embed_chunks(
        chunks=chunks,
        model=model,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    document_embedding_ms = (
        perf_counter() - embedding_started
    ) * 1000

    points = [
        embedded_chunk_to_point(embedded_chunk)
        for embedded_chunk in embedded_chunks
    ]

    indexing_started = perf_counter()
    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )
    indexing_ms = (perf_counter() - indexing_started) * 1000

    return {
        "chunk_count": len(chunks),
        "chunking_ms": round(chunking_ms, 3),
        "document_embedding_ms": round(document_embedding_ms, 3),
        "indexing_ms": round(indexing_ms, 3),
    }


def prepare_query_vectors(questions: list[dict], model) -> dict[str, list]:
    vectors = {}

    for index, item in enumerate(questions, start=1):
        vectors[item["id"]] = embed_query(
            query=item["question"],
            model=model,
        )
        print(
            f"Sorgu vektörü hazırlandı: "
            f"{index:02d}/{len(questions)} {item['id']}"
        )

    return vectors


def search_collection(
    client,
    collection_name: str,
    query_vector: list[float],
    score_threshold: float,
) -> tuple[list, float]:
    started = perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=MAX_TOP_K,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )
    search_ms = (perf_counter() - started) * 1000

    return response.points, search_ms


def evaluate_configuration(
    client,
    collection_name: str,
    questions: list[dict],
    query_vectors: dict[str, list],
) -> dict:
    answerable_count = 0
    no_answer_count = 0
    rejected_no_answer_count = 0
    hit_counts = {top_k: 0 for top_k in TOP_K_VALUES}
    reciprocal_ranks = []
    search_latencies_ms = []
    question_results = []

    for item in questions:
        answerable = item.get("answerable", True)
        threshold = (
            RANKING_SCORE_THRESHOLD
            if answerable
            else NO_ANSWER_SCORE_THRESHOLD
        )
        results, search_ms = search_collection(
            client=client,
            collection_name=collection_name,
            query_vector=query_vectors[item["id"]],
            score_threshold=threshold,
        )
        search_latencies_ms.append(search_ms)

        record = {
            "id": item["id"],
            "answerable": answerable,
            "search_ms": round(search_ms, 3),
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

            record["first_relevant_rank"] = first_relevant_rank

            for top_k in TOP_K_VALUES:
                hit = (
                    first_relevant_rank is not None
                    and first_relevant_rank <= top_k
                )
                hit_counts[top_k] += int(hit)
        else:
            no_answer_count += 1
            rejected = not results
            rejected_no_answer_count += int(rejected)
            record["no_answer_rejected"] = rejected
            record["top_score"] = (
                round(float(results[0].score), 6)
                if results
                else None
            )

        question_results.append(record)

    return {
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
        "average_search_ms": round(mean(search_latencies_ms), 3),
        "questions": question_results,
    }


def select_best_configuration(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda item: (
            item["metrics"]["recall_at_3"],
            item["metrics"]["mrr"],
            item["metrics"]["recall_at_1"],
            -item["metrics"]["average_search_ms"],
        ),
    )


def print_comparison(results: list[dict], best: dict) -> None:
    print("\n" + "=" * 76)
    print("CHUNK YAPILANDIRMASI KARŞILAŞTIRMASI")
    print("=" * 76)

    for result in results:
        config = result["configuration"]
        metrics = result["metrics"]
        print(
            f"{config['chunk_size']:>4}/{config['overlap']:<3} | "
            f"chunk: {result['indexing']['chunk_count']:>3} | "
            f"R@1: {metrics['recall_at_1']:.2%} | "
            f"R@3: {metrics['recall_at_3']:.2%} | "
            f"R@5: {metrics['recall_at_5']:.2%} | "
            f"MRR: {metrics['mrr']:.4f} | "
            f"arama: {metrics['average_search_ms']:.2f} ms"
        )

    best_config = best["configuration"]
    print(
        "\nSeçilen yapılandırma: "
        f"chunk_size={best_config['chunk_size']}, "
        f"overlap={best_config['overlap']}"
    )
    print("Sonuç dosyası:", RESULTS_PATH)


def run_experiments() -> dict:
    dataset, questions = load_questions()

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Deney dokümanı bulunamadı: {DOCUMENT_PATH}"
        )

    documents = load_document(DOCUMENT_PATH)
    model, _ = load_embedding_model()
    query_vectors = prepare_query_vectors(questions, model)
    client = create_qdrant_client()
    experiment_results = []

    for configuration in EXPERIMENTS:
        collection_name = build_collection_name(configuration)
        print(
            "\nDeney başlatıldı: "
            f"chunk_size={configuration['chunk_size']}, "
            f"overlap={configuration['overlap']}"
        )

        create_experiment_collection(client, collection_name)

        try:
            indexing = index_configuration(
                client=client,
                collection_name=collection_name,
                documents=documents,
                model=model,
                configuration=configuration,
            )
            metrics = evaluate_configuration(
                client=client,
                collection_name=collection_name,
                questions=questions,
                query_vectors=query_vectors,
            )
            experiment_results.append(
                {
                    "configuration": configuration,
                    "indexing": indexing,
                    "metrics": metrics,
                }
            )
        finally:
            if client.collection_exists(collection_name):
                client.delete_collection(collection_name)

    best = select_best_configuration(experiment_results)
    output = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "document": str(DOCUMENT_PATH.relative_to(PROJECT_ROOT)),
        "selection_rule": (
            "Recall@3, MRR, Recall@1, then lower search latency"
        ),
        "experiments": experiment_results,
        "best_configuration": best["configuration"],
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print_comparison(experiment_results, best)
    return output


if __name__ == "__main__":
    run_experiments()
