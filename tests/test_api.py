from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api


# Sağlık kontrolleri için hazır bir sahte Qdrant istemcisi oluşturur.
@pytest.fixture
def qdrant_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


# API testleri için bağımlılıkları ayarlanmış bir test istemcisi sağlar.
@pytest.fixture
def client(
    qdrant_client: MagicMock,
):
    api.app.state.embedding_model = object()
    api.app.state.qdrant_client = qdrant_client

    test_client = TestClient(api.app)

    yield test_client

    test_client.close()


# Kök uç noktanın API bağlantılarını döndürdüğünü doğrular.
def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "RAG Document QA API",
        "documentation": "/docs",
        "health": "/health",
    }


# Tüm bağımlılıklar hazırken sağlık uç noktasının başarılı olduğunu doğrular.
def test_health_endpoint_is_ready(
    client: TestClient,
    qdrant_client: MagicMock,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api,
        "is_model_available",
        lambda model_name: True,
    )

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["collection_ready"] is True
    assert data["embedding_model_loaded"] is True
    assert data["ollama_ready"] is True
    assert data["llm_model"] == api.MODEL_NAME

    qdrant_client.collection_exists.assert_called_once_with(
        api.COLLECTION_NAME
    )


# Qdrant koleksiyonu yokken sağlık uç noktasının 503 döndürdüğünü doğrular.
def test_health_returns_503_when_collection_is_missing(
    client: TestClient,
    qdrant_client: MagicMock,
) -> None:
    qdrant_client.collection_exists.return_value = False

    response = client.get("/health")

    assert response.status_code == 503
    assert "collection hazır değil" in response.json()["detail"]


# Qdrant'a ulaşılamadığında sağlık uç noktasının 503 döndürdüğünü doğrular.
def test_health_returns_503_when_qdrant_is_unreachable(
    client: TestClient,
    qdrant_client: MagicMock,
) -> None:
    qdrant_client.collection_exists.side_effect = (
        RuntimeError("Bağlantı hatası")
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Qdrant servisine ulaşılamadı."
    )


# Ollama modeli yokken sağlık uç noktasının 503 döndürdüğünü doğrular.
def test_health_returns_503_when_ollama_model_is_missing(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api,
        "is_model_available",
        lambda model_name: False,
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert "Ollama modeli hazır değil" in (
        response.json()["detail"]
    )


# Doküman uç noktasının indeks özetini doğru döndürdüğünü doğrular.
def test_documents_endpoint(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api,
        "list_indexed_documents",
        lambda qdrant_client: [
            {
                "document_id": "doc-1",
                "document_name": "uyku.pdf",
                "document_hash": "hash-1",
                "chunk_count": 23,
            }
        ],
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "total_documents": 1,
        "total_chunks": 23,
        "documents": [
            {
                "document_id": "doc-1",
                "document_name": "uyku.pdf",
                "document_hash": "hash-1",
                "chunk_count": 23,
            }
        ],
    }


# Doküman listeleme hatasının 500 yanıtına dönüştürüldüğünü doğrular.
def test_documents_returns_500_on_unexpected_error(
    client: TestClient,
    monkeypatch,
) -> None:
    # Qdrant listeleme hatasını taklit eder.
    def raise_error(qdrant_client):
        raise RuntimeError("Qdrant hatası")

    monkeypatch.setattr(
        api,
        "list_indexed_documents",
        raise_error,
    )

    response = client.get("/documents")

    assert response.status_code == 500
    assert "beklenmeyen bir hata" in (
        response.json()["detail"]
    )


# Soru uç noktalarının RAG cevabını beklenen biçimde döndürdüğünü doğrular.
@pytest.mark.parametrize("endpoint", ["/query", "/ask"])
def test_query_endpoint_returns_rag_answer(
    client: TestClient,
    monkeypatch,
    endpoint: str,
) -> None:
    # RAG hattının başarılı örnek cevabını taklit eder.
    def fake_answer_question(**options):
        assert options["question"] == (
            "Uyku neden önemlidir?"
        )

        return {
            "question": options["question"],
            "answer": "Uyku, sağlığın korunmasına yardımcı olur.",
            "sources": [
                {
                    "source_number": 1,
                    "document_name": "uyku.pdf",
                    "page": 1,
                    "chunk_index": 1,
                    "chunk_id": "doc-1_1_1",
                    "score": 0.91,
                    "text": "Uyku sağlığı destekler.",
                }
            ],
            "retrieval_ms": 12.5,
            "generation_seconds": 1.25,
        }

    monkeypatch.setattr(
        api,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        endpoint,
        json={
            "question": "  Uyku neden önemlidir?  "
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "Uyku neden önemlidir?"
    assert data["sources"][0]["page"] == 1
    assert data["retrieval_latency_ms"] == 12.5
    assert data["generation_latency_ms"] == 1250.0


# Boş veya yalnızca boşluk içeren soruların reddedildiğini doğrular.
@pytest.mark.parametrize(
    "question",
    ["", "   "],
)
def test_query_rejects_blank_question(
    client: TestClient,
    question: str,
) -> None:
    response = client.post(
        "/query",
        json={"question": question},
    )

    assert response.status_code == 422


# Soru alanı bulunmayan isteklerin reddedildiğini doğrular.
def test_query_rejects_missing_question(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={},
    )

    assert response.status_code == 422


# Uzunluk sınırını aşan soruların reddedildiğini doğrular.
def test_query_rejects_question_longer_than_limit(
    client: TestClient,
) -> None:
    response = client.post(
        "/query",
        json={"question": "a" * 501},
    )

    assert response.status_code == 422


# RAG doğrulama hatasının 400 yanıtına dönüştürüldüğünü doğrular.
def test_query_returns_400_for_value_error(
    client: TestClient,
    monkeypatch,
) -> None:
    # RAG doğrulama hatasını taklit eder.
    def raise_value_error(**options):
        raise ValueError("Geçersiz soru")

    monkeypatch.setattr(
        api,
        "answer_question",
        raise_value_error,
    )

    response = client.post(
        "/query",
        json={"question": "Örnek soru"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Geçersiz soru"


# Beklenmeyen RAG hatasının 500 yanıtına dönüştürüldüğünü doğrular.
def test_query_returns_500_for_unexpected_error(
    client: TestClient,
    monkeypatch,
) -> None:
    # Beklenmeyen RAG çalışma zamanı hatasını taklit eder.
    def raise_runtime_error(**options):
        raise RuntimeError("Beklenmeyen hata")

    monkeypatch.setattr(
        api,
        "answer_question",
        raise_runtime_error,
    )

    response = client.post(
        "/query",
        json={"question": "Örnek soru"},
    )

    assert response.status_code == 500
    assert "beklenmeyen" in (
        response.json()["detail"].lower()
    )

def test_response_contains_total_latency_header(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "x-process-time-ms" in response.headers
    assert float(
        response.headers["x-process-time-ms"]
    ) >= 0