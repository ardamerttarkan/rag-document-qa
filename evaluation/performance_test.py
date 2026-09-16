import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "performance_metrics.json"
)

API_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{API_URL}/query"
HEALTH_ENDPOINT = f"{API_URL}/health"

# PDF'nin beş farklı sayfasından birer soru.
QUESTION_IDS = [
    "Q001",
    "Q010",
    "Q019",
    "Q028",
    "Q037",
]

REQUEST_TIMEOUT_SECONDS = 120.0


def load_selected_questions() -> list[dict]:
    data = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    questions = {
        item["id"]: item
        for item in data["questions"]
    }

    missing_ids = [
        question_id
        for question_id in QUESTION_IDS
        if question_id not in questions
    ]

    if missing_ids:
        raise ValueError(
            "questions.json içerisinde bulunamayan "
            f"soru kimlikleri: {missing_ids}"
        )

    return [
        questions[question_id]
        for question_id in QUESTION_IDS
    ]


def summarize_latency(
    values: list[float],
) -> dict | None:
    if not values:
        return None

    return {
        "minimum": round(min(values), 2),
        "average": round(mean(values), 2),
        "median": round(median(values), 2),
        "maximum": round(max(values), 2),
    }


def run_performance_test() -> dict:
    questions = load_selected_questions()
    request_results = []

    with httpx.Client(
        timeout=REQUEST_TIMEOUT_SECONDS
    ) as client:
        health_response = client.get(
            HEALTH_ENDPOINT
        )
        health_response.raise_for_status()

        for index, item in enumerate(
            questions,
            start=1,
        ):
            start_time = perf_counter()

            try:
                response = client.post(
                    QUERY_ENDPOINT,
                    json={
                        "question": item["question"]
                    },
                )

                client_total_latency_ms = round(
                    (
                        perf_counter()
                        - start_time
                    )
                    * 1000,
                    2,
                )

                server_latency_header = (
                    response.headers.get(
                        "x-process-time-ms"
                    )
                )

                server_total_latency_ms = (
                    float(server_latency_header)
                    if server_latency_header
                    else None
                )

                if response.status_code == 200:
                    payload = response.json()

                    result = {
                        "id": item["id"],
                        "question": item["question"],
                        "success": True,
                        "status_code": (
                            response.status_code
                        ),
                        "client_total_latency_ms": (
                            client_total_latency_ms
                        ),
                        "server_total_latency_ms": (
                            server_total_latency_ms
                        ),
                        "retrieval_latency_ms": (
                            payload[
                                "retrieval_latency_ms"
                            ]
                        ),
                        "generation_latency_ms": (
                            payload[
                                "generation_latency_ms"
                            ]
                        ),
                        "source_count": len(
                            payload["sources"]
                        ),
                        "error": None,
                    }
                else:
                    result = {
                        "id": item["id"],
                        "question": item["question"],
                        "success": False,
                        "status_code": (
                            response.status_code
                        ),
                        "client_total_latency_ms": (
                            client_total_latency_ms
                        ),
                        "server_total_latency_ms": (
                            server_total_latency_ms
                        ),
                        "retrieval_latency_ms": None,
                        "generation_latency_ms": None,
                        "source_count": None,
                        "error": response.text,
                    }

            except httpx.RequestError as error:
                client_total_latency_ms = round(
                    (
                        perf_counter()
                        - start_time
                    )
                    * 1000,
                    2,
                )

                result = {
                    "id": item["id"],
                    "question": item["question"],
                    "success": False,
                    "status_code": None,
                    "client_total_latency_ms": (
                        client_total_latency_ms
                    ),
                    "server_total_latency_ms": None,
                    "retrieval_latency_ms": None,
                    "generation_latency_ms": None,
                    "source_count": None,
                    "error": str(error),
                }

            request_results.append(result)

            status = (
                "başarılı"
                if result["success"]
                else "başarısız"
            )

            print(
                f"[{index:02d}/{len(questions):02d}] "
                f"{item['id']} | {status} | "
                f"toplam: "
                f"{result['client_total_latency_ms']}"
                " ms"
            )

    successful_results = [
        item
        for item in request_results
        if item["success"]
    ]

    client_latencies = [
        item["client_total_latency_ms"]
        for item in successful_results
    ]

    server_latencies = [
        item["server_total_latency_ms"]
        for item in successful_results
        if item["server_total_latency_ms"]
        is not None
    ]

    retrieval_latencies = [
        item["retrieval_latency_ms"]
        for item in successful_results
    ]

    generation_latencies = [
        item["generation_latency_ms"]
        for item in successful_results
    ]

    success_count = len(successful_results)
    request_count = len(request_results)

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_url": API_URL,
        "question_ids": QUESTION_IDS,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": (
            request_count - success_count
        ),
        "success_rate_percent": round(
            success_count
            / request_count
            * 100,
            2,
        ),
        "latency_summary_ms": {
            "client_total": summarize_latency(
                client_latencies
            ),
            "server_total": summarize_latency(
                server_latencies
            ),
            "retrieval": summarize_latency(
                retrieval_latencies
            ),
            "generation": summarize_latency(
                generation_latencies
            ),
        },
        "requests": request_results,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report


if __name__ == "__main__":
    report = run_performance_test()

    print("\n" + "=" * 55)
    print("PERFORMANS RAPORU")
    print("=" * 55)
    print(
        "İstek sayısı:",
        report["request_count"],
    )
    print(
        "Başarı oranı:",
        f"{report['success_rate_percent']:.2f}%",
    )

    summaries = report["latency_summary_ms"]

    for name, values in summaries.items():
        if values is None:
            continue

        print(
            f"{name:15} | "
            f"ortalama: {values['average']:.2f} ms | "
            f"min: {values['minimum']:.2f} ms | "
            f"max: {values['maximum']:.2f} ms"
        )

    print("Sonuç dosyası:", RESULT_PATH)