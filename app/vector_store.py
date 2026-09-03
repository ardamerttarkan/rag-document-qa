from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from chunking import chunk_documents
from embeddings import (
    DEFAULT_BATCH_SIZE,
    embed_chunks,
    load_embedding_model,
)
from ingest import PROJECT_ROOT, load_document



QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "rag_documents"
EMBEDDING_DIMENSION = 384


# Qdrant veritabanı istemcisini oluşturur.
def create_qdrant_client() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)
    return client


# Gerekli Qdrant koleksiyonunun var olmasını sağlar.
def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print("Collection zaten mevcut:", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    print("Collection oluşturuldu:", COLLECTION_NAME)


# Metin parçası için benzersiz bir nokta kimliği üretir.
def create_point_id(chunk_id: str) -> str:
    point_id = uuid5(
        NAMESPACE_URL,
        f"rag-document-qa/{chunk_id}",
    )

    return str(point_id)


# Embedding içeren metin parçasını Qdrant noktasına dönüştürür.
def embedded_chunk_to_point(
    embedded_chunk: dict,
) -> models.PointStruct:
    vector = embedded_chunk["embedding"]
    metadata = embedded_chunk["metadata"].copy()
    chunk_id = metadata["chunk_id"]

    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"{chunk_id} için beklenen vektör boyutu "
            f"{EMBEDDING_DIMENSION}, gelen {len(vector)}"
        )

    payload = metadata
    payload["text"] = embedded_chunk["text"]

    point = models.PointStruct(
        id=create_point_id(chunk_id),
        vector=vector,
        payload=payload,
    )

    return point


# Embedding içeren metin parçalarını Qdrant'a kaydeder.
def upsert_embedded_chunks(
    client: QdrantClient,
    embedded_chunks: list[dict],
) -> int:
    if not embedded_chunks:
        raise ValueError("Qdrant'a kaydedilecek chunk bulunamadı.")

    points = []

    for embedded_chunk in embedded_chunks:
        point = embedded_chunk_to_point(embedded_chunk)
        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)

def list_indexed_documents(
    client: QdrantClient,
) -> list[dict]:
    documents: dict[str, dict] = {}
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            document_id = payload.get("document_id")

            if not document_id:
                continue

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "document_name": payload.get(
                        "document_name"
                    ),
                    "document_hash": payload.get(
                        "document_hash"
                    ),
                    "chunk_count": 0,
                }

            documents[document_id][
                "chunk_count"
            ] += 1

        if next_offset is None:
            break

    return sorted(
        documents.values(),
        key=lambda document: (
            document.get("document_name") or ""
        ).lower(),
    )

if __name__ == "__main__":
    file_path = (
        PROJECT_ROOT
        / "pdf_samples"
        / "uyku_duzeni.pdf"
    )

    documents = load_document(file_path)

    chunks = chunk_documents(
        documents=documents,
        chunk_size=500,
        overlap=100,
    )

    print("Toplam chunk sayısı:", len(chunks))

    model, model_load_duration = load_embedding_model()

    embedded_chunks, embedding_duration = embed_chunks(
        chunks=chunks,
        model=model,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    qdrant_client = create_qdrant_client()
    ensure_collection(qdrant_client)

    upserted_count = upsert_embedded_chunks(
        client=qdrant_client,
        embedded_chunks=embedded_chunks,
    )

    stored_count = qdrant_client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    first_chunk = embedded_chunks[0]
    first_chunk_id = first_chunk["metadata"]["chunk_id"]

    print("Qdrant'a gönderilen point:", upserted_count)
    print("Collection içindeki point:", stored_count)
    print("İlk chunk ID:", first_chunk_id)
    print(
        "İlk Qdrant point ID:",
        create_point_id(first_chunk_id),
    )
