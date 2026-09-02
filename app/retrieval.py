from math import isfinite

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from embeddings import (
    EMBEDDING_DIMENSION,
    load_embedding_model,
)
from vector_store import (
    COLLECTION_NAME,
    create_qdrant_client,
)


DEFAULT_TOP_K = 3
DEFAULT_SCORE_THRESHOLD = 0.80


# Kullanıcı sorgusunu sayısal bir vektöre dönüştürür.
def embed_query(
    query: str,
    model: SentenceTransformer,
) -> list[float]:
    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Kullanıcı sorusu boş olamaz.")

    query_text = f"query: {cleaned_query}"

    vector = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vector_list = vector.tolist()

    if len(vector_list) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Beklenen query vektör boyutu "
            f"{EMBEDDING_DIMENSION}, "
            f"gelen {len(vector_list)}"
        )

    if not all(isfinite(value) for value in vector_list):
        raise ValueError(
            "Query embedding içerisinde geçersiz sayı bulundu."
        )

    return vector_list


# Sorguya en çok benzeyen metin parçalarını arar.
def search_similar_chunks(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list:
    if top_k <= 0:
        raise ValueError("Top-K sıfırdan büyük olmalıdır.")

    if not -1 <= score_threshold <= 1:
        raise ValueError(
            "Benzerlik eşiği -1 ile 1 arasında olmalıdır."
        )

    if len(query_vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Query vektörü {EMBEDDING_DIMENSION} "
            "boyutunda olmalıdır."
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False,
    )

    return response.points

if __name__ == "__main__":
    query = input("Sorunuzu yazın: ").strip()

    model, model_load_duration = load_embedding_model()

    query_vector = embed_query(
        query=query,
        model=model,
    )

    qdrant_client = create_qdrant_client()

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection bulunamadı: {COLLECTION_NAME}"
        )

    results = search_similar_chunks(
    client=qdrant_client,
    query_vector=query_vector,
    top_k=DEFAULT_TOP_K,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
)

    print("Kullanıcı sorusu:", query)
    print("Top-K:", DEFAULT_TOP_K)
    print("Bulunan sonuç sayısı:", len(results))
    
    if not results:
        print(
        "Dokümanlarda soruyla yeterince ilgili "
        "bir bilgi bulunamadı."
    )

    for result_number, result in enumerate(
        results,
        start=1,
    ):
        payload = result.payload or {}

        print("-" * 50)
        print("Sonuç sırası:", result_number)
        print(f"Benzerlik skoru: {result.score:.4f}")
        print("Doküman:", payload.get("document_name"))
        print("Sayfa:", payload.get("page"))
        print("Chunk sırası:", payload.get("chunk_index"))
        print("Chunk ID:", payload.get("chunk_id"))
        print("Metin:")
        print(payload.get("text"))
